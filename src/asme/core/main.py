import json
import os
import shutil
from pathlib import Path
from typing import Callable
from typing import List
from typing import Optional

import optuna
import torch
import typer

from asme.core.callbacks.metrics_history import MetricsHistoryCallback
from asme.core.init.config import Config
from asme.core.init.config_keys import CHECKPOINT_CONFIG_DIR_PATH
from asme.core.init.config_keys import CHECKPOINT_CONFIG_KEY
from asme.core.init.config_keys import TRAINER_CONFIG_KEY
from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.factories.metrics.metrics_container import MetricsContainerFactory
from asme.core.init.templating.search.configuration import SearchConfigurationTemplateProcessor
from asme.core.init.templating.search.processor import SearchTemplateProcessor
from asme.core.init.templating.search.resolver import OptunaParameterResolver
from asme.core.model_summary import ModelSummary
from asme.core.utils import ioutils
from asme.core.utils.ioutils import determine_log_dir
from asme.core.utils.ioutils import finished_flag_exists
from asme.core.utils.ioutils import save_config
from asme.core.utils.ioutils import save_finished_flag
from asme.core.utils.pred_utils import move_tensors_to_device
from asme.core.utils.run_utils import create_container
from asme.core.utils.run_utils import load_and_restore_from_file_or_study
from asme.core.utils.run_utils import load_config
from asme.core.utils.run_utils import load_config_from_json
from asme.core.utils.run_utils import load_hyperopt_config
from asme.core.utils.run_utils import log_dataloader_example
from asme.core.utils.run_utils import OBJECTIVE_METRIC_KEY
from asme.core.utils.run_utils import TRIAL_BASE_PATH
from asme.core.writer.results.results_writer import build_result_writer
from asme.core.writer.results.results_writer import check_file_format_supported
from jinja2 import Template
from loguru import logger
from optuna.study import StudyDirection
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LoggerCollection
from pytorch_lightning.loggers import MLFlowLogger
from tqdm import tqdm

_ERROR_MESSAGE_LOAD_CHECKPOINT_FROM_FILE_OR_STUDY = "You have to specify at least the checkpoint file and config or" \
                                                    " the study name and study storage to infer the config and " \
                                                    "checkpoint path"

# (AD) for now this is necessary to prevent segfaults occurring in conjunction with using multiple workers and the AimLogger
# see https://github.com/aimhubio/aim/issues/1297
torch.multiprocessing.set_start_method('spawn', force=True)

app = typer.Typer()


@app.command()
def train(config_file: Path = typer.Argument(..., help='the path to the config file', exists=True),
          do_resume: bool = typer.Option(False, help='flag iff the model should resume training from a checkpoint'),
          print_train_val_examples: bool = typer.Option(False, help='print examples of the training '
                                                                    'and evaluation dataset before starting training'),
          seed: Optional[int] = typer.Option(default=42, help='seed used eg for the sampled evaluation')
          ) -> None:


    if seed is not None:
        seed_everything(seed)

    config_file_path = Path(config_file)
    config = load_config(config_file_path)

    container = create_container(config)
    trainer = container.trainer().build()



    # save plain json config to the log dir/root dir of the trainer
    log_dir = determine_log_dir(trainer)
    save_config(config, config_file, log_dir)

    if do_resume:
        resume(log_dir, None)

    else:
        train_dataloader = container.train_dataloader()
        validation_dataloader = container.validation_dataloader()

        if print_train_val_examples:
            tokenizers = container.tokenizers()
            log_dataloader_example(train_dataloader, tokenizers, 'training')
            log_dataloader_example(validation_dataloader, tokenizers, 'validation')

        module = container.module()

        # Save the hyperparameters of the dataset preprocessing
        #preprocessing_parameters = container.datamodule().config.preprocessing_config_values
        #if preprocessing_parameters is not None:
        #    preprocessing_parameters = {f"datamodule/{param}": value for param, value in preprocessing_parameters.items()}
        #    module.save_hyperparameters(preprocessing_parameters)

        parameters = list(module.parameters())
        trainable_parameters = [p for p in parameters if p.requires_grad]
        unique = {p.data_ptr(): p for p in parameters}.values()
        print("Unique trainable parameters",sum(p.numel() for p in unique))




        print(ModelSummary(module,max_depth=-1))

        trainer.fit(module,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=validation_dataloader)

        save_finished_flag(log_dir)












@app.command()
def search(template_file: Path = typer.Argument(..., help='the path to the config file'),
           optimization_config_file: Path = typer.Argument(..., help='path to a file containing optimization specs.'),
           study_name: str = typer.Argument(..., help='the study name of an existing optuna study'),
           study_storage: str = typer.Argument(..., help='the connection string for the study storage'),

           objective_metric: str = typer.Argument(..., help='the name of the metric to watch during the study'
                                                            '(e.g. recall@5).'),
           study_direction: str = typer.Option(default="maximize", help="minimize / maximize"),
           num_trials: int = typer.Option(default=20, help='the number of trials to execute')
           ) -> None:

    # check if objective_metric is defined
    test_config = load_config(template_file)
    metrics_factory = MetricsContainerFactory()

    build_context = BuildContext(test_config, Context())
    build_context.enter_sections(["module","metrics"])
    #build_context.leave_sections(sub_section)
    test_metrics_container = metrics_factory.build(build_context)
    test = test_metrics_container.get_metric_names()
    if objective_metric not in test_metrics_container.get_metric_names():
        raise ValueError(f'{objective_metric} not configured. '
                         f'Can not optimize hyperparameters using the specified objective')

    def config_from_template(template_file: Path,
                             hyperopt_config_file: Path,
                             trial: optuna.Trial) -> Config:
        """
        Loads the model template and hyperopt config files. Fully resolves the model configuration and patches the model
        configuration with the resolved hyper parameters from the study.
        Both the final hyper optimization and model config are written into the output directory for later reference.

        :param template_file: a config file template path.
        :param hyperopt_config_file: a config file with hyper parameter optimiziation configs.
        :param trial: a trial object.
        :return: the final configuration with resolved hyperparameters according to the hyperopt config file.
        """
        config = load_config(template_file,
                             additional_tail_processors=[SearchConfigurationTemplateProcessor(trial)])
        hyperopt_config = load_hyperopt_config(hyperopt_config_file,
                                               [SearchTemplateProcessor(OptunaParameterResolver(trial))])

        # infer model train directory from checkpoint output path set via `SearchConfigurationTemplateProcessor`.
        output_path = Path(config.get([TRAINER_CONFIG_KEY, CHECKPOINT_CONFIG_KEY, CHECKPOINT_CONFIG_DIR_PATH])).parent

        if not output_path.exists():
            output_path.mkdir(parents=True)

        shutil.copy2(hyperopt_config_file, output_path / "hyperopt_study.jsonnet")
        with (output_path / "hyperopt_trial.jsonnet").open("w") as hyperopt_config_file:
            json.dump(hyperopt_config.config, hyperopt_config_file, indent=2)

        patched_config = config.patch(hyperopt_config)

        # FIXME: adhoc fix to make output_path available in the configuration, remove when this is common in a configuration.
        patched_config.set_if_absent(["output_path"], str(output_path))
        return patched_config

    study = optuna.create_study(study_name=study_name,
                                storage=study_storage,
                                load_if_exists=True,
                                direction=study_direction)
    study.set_user_attr(OBJECTIVE_METRIC_KEY, objective_metric)

    for trial_run in range(1, num_trials + 1):
        print(f"Running trial {trial_run} / {num_trials}")
        trial = study.ask()

        trial_study_direction = trial.study.direction
        objective_best = {
            StudyDirection.MINIMIZE: min,
            StudyDirection.MAXIMIZE: max
        }[trial_study_direction]

        # load config template, apply processors and write to `tmp_config_file`
        model_config = config_from_template(template_file, optimization_config_file, trial)

        metrics_tracker = MetricsHistoryCallback()

        container = create_container(model_config)

        module = container.module()
        trainer_builder = container.trainer()
        trainer_builder.add_callback(metrics_tracker)

        trainer = trainer_builder.build()
        log_dir = determine_log_dir(trainer)
        trial.set_user_attr(TRIAL_BASE_PATH, str(log_dir))

        # store mlflow run id
        def save_mlflow_id(trainer: Trainer, log_dir: Path):
            if trainer.logger is None:
                return

            if type(trainer.logger) is MLFlowLogger:
                id_file_path = log_dir / "mlflow_run_id.txt"
                with id_file_path.open("w") as id_file:
                    version = trainer.logger.version
                    experiment = trainer.logger.name

                    id_file.write(f"experiment:{experiment}\nversion:{version}")

            if type(trainer.logger) is LoggerCollection:
                for l in trainer.logger._logger_iterable:
                    if type(l) is MLFlowLogger:
                        id_file_path = log_dir / "mlflow_run_id.txt"
                        with id_file_path.open("w") as id_file:
                            version = trainer.logger.version
                            experiment = trainer.logger.name

                            id_file.write(f"experiment:{experiment}\nversion:{version}")

        # save config of current run to its log dir
        save_config(model_config, template_file, log_dir)
        save_mlflow_id(trainer, log_dir)

        trainer.fit(
            model=module,
            train_dataloaders=container.train_dataloader(),
            val_dataloaders=container.validation_dataloader()
        )
        save_finished_flag(log_dir)

        def _find_best_value(key: str, best: Callable[[List[float]], float] = min) -> float:
            values = [history_entry[key] for history_entry in metrics_tracker.metric_history]
            return best(values)

        study.tell(trial, _find_best_value(objective_metric, objective_best))


@app.command()
def predict(output_file: Path = typer.Argument(..., help='path where output is written'),
            gpu: Optional[int] = typer.Option(default=0, help='number of gpus to use.'),
            checkpoint_file: Path = typer.Option(default=None, help='path to the checkpoint file'),
            config_file: Path = typer.Option(default=None, help='the path to the config file'),
            study_name: str = typer.Option(default=None, help='the study name of an existing study'),
            study_storage: str = typer.Option(default=None, help='the connection string for the study storage'),
            overwrite: Optional[bool] = typer.Option(default=False, help='overwrite output file if it exists.'),
            seed: Optional[int] = typer.Option(default=None, help='seed used eg for the sampled evaluation'),
            ):
    # checking if the file already exists
    if not overwrite and output_file.exists():
        logger.error(f"${output_file} already exists. If you want to overwrite it, use `--overwrite`.")
        exit(2)

    container = load_and_restore_from_file_or_study(checkpoint_file, config_file, study_name, study_storage,
                                                    gpus=gpu)
    if container is None:
        logger.error(_ERROR_MESSAGE_LOAD_CHECKPOINT_FROM_FILE_OR_STUDY)
        exit(-1)

    if seed is not None:
        seed_everything(seed)

    module = container.module()
    trainer = container.trainer().build()
    test_loader = container.test_dataloader()
    writer = container.writer()


    module.eval()
    device = torch.device('cuda') if torch.cuda.is_available() and gpu > 0 else torch.device('cpu')
    module.to(device)
    with torch.no_grad():
        with open(output_file, 'w') as result_file:
            writer.init_file(file_handle=result_file)
            for batch_index, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
                batch = move_tensors_to_device(batch, device)
                logits = module.predict_step(batch=batch, batch_idx=batch_index)
                writer.write_evaluation(batch_index, batch, logits)


@app.command()
def evaluate(config_file: Path = typer.Option(default=None, help='the path to the config file'),
             checkpoint_file: Path = typer.Option(default=None, help='path to the checkpoint file'),
             study_name: str = typer.Option(default=None, help='the study name of an existing study'),
             study_storage: str = typer.Option(default=None, help='the connection string for the study storage'),
             output_file: Optional[Path] = typer.Option(default=None, help='path where output is written'),
             gpu: Optional[int] = typer.Option(default=0, help='number of gpus to use.'),
             overwrite: Optional[bool] = typer.Option(default=False, help='overwrite output file if it exists.'),
             seed: Optional[int] = typer.Option(default=None, help='seed used eg for the sampled evaluation')
             ):
    write_results_to_file = output_file is not None
    if write_results_to_file and not overwrite and output_file.exists():
        logger.error(f"${output_file} already exists. If you want to overwrite it, use `--overwrite`.")
        exit(-1)

    if write_results_to_file and not check_file_format_supported(output_file):
        logger.error(f"'{output_file.suffix} is not a supported format to write predictions to a file.'")
        exit(-1)

    if write_results_to_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        seed_everything(seed)

    container = load_and_restore_from_file_or_study(checkpoint_file, config_file, study_name, study_storage,
                                                    gpus=gpu)
    if container is None:
        logger.error(_ERROR_MESSAGE_LOAD_CHECKPOINT_FROM_FILE_OR_STUDY)
        exit(-1)

    module = container.module()
    trainer = container.trainer().build()
    test_loader = container.test_dataloader()

    eval_results = trainer.test(module, dataloaders=test_loader, verbose=not write_results_to_file)

    if write_results_to_file:
        with open(output_file, 'w') as output_file_handle:
            result_writer = build_result_writer(output_file_handle)
            # FIXME: currently we have for every model a corresponding module
            # get the first eval results, only one test_dataloader was provided
            result_writer.write_overall_results(type(module).__name__, eval_results[0])


@app.command()
def resume(log_dir: str = typer.Argument(..., help='the path to the logging directory of the run to be resumed'),
           checkpoint_file: Optional[str] = typer.Option(default=None,
                                                         help="the name of the checkpoint file to resume from")):
    log_dir = Path(log_dir)

    # check for finished flag
    if finished_flag_exists(log_dir):
        logger.error(f"Found a finished flag in '{log_dir}'. Training has finished and will not be resumed.")
        exit(-1)

    # check for config file
    #config_file = log_dir / ioutils.PROCESSED_CONFIG_NAME
    config_file = log_dir / f"{ioutils.PROCESSED_CONFIG_NAME}.yaml"  # TODO: bug, missing config file suffix
    if not os.path.isfile(config_file):
        logger.error(f"Could not find '{ioutils.PROCESSED_CONFIG_NAME} in path {log_dir}.")
        exit(-1)

    raw_config = load_config(Path(config_file))
    # determine checkpoint dir:
    checkpoint_dir = Path(
        raw_config.get_config(["trainer", "checkpoint"]).get_or_default("dirpath", log_dir / "checkpoints"))
    # if no checkpoint file is provided we use the last checkpoint.
    if checkpoint_file is None:
        checkpoint_file = "last.ckpt"

    checkpoint_path = checkpoint_dir / checkpoint_file
    if not os.path.isfile(checkpoint_path):
        logger.error("Could not determine the last checkpoint. "
                     "You can specify a particular checkpoint via the --checkpoint-file option.")
        train(Path(config_file), do_resume=False)

    container = create_container(raw_config)
    module = container.module()

    trainer_builder = container.trainer()
    trainer = trainer_builder.from_checkpoint(checkpoint_path).build()

    train_loader = container.train_dataloader()
    validation_loader = container.validation_dataloader()

    #trainer.fit(module, train_dataloader=train_loader, val_dataloaders=validation_loader)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=validation_loader)  # TODO: bug, typo

    # Save finished flag
    save_finished_flag(log_dir)

if __name__ == "__main__":
    app()
