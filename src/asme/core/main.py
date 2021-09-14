import os
import shutil
from contextlib import contextmanager, redirect_stdout

import numpy as np
import torch
import typer
import optuna
import json
from pathlib import Path
from typing import Optional, Callable, List, Iterator, Tuple

from pytorch_lightning.loggers import LoggerCollection, MLFlowLogger

from asme.core.init.templating.search.resolver import OptunaParameterResolver

from asme.core.init.templating.search.processor import SearchTemplateProcessor

from asme.core.init.config import Config

from asme.core.init.config_keys import TRAINER_CONFIG_KEY, CHECKPOINT_CONFIG_KEY, CHECKPOINT_CONFIG_DIR_PATH
from optuna.study import StudyDirection
from pytorch_lightning import seed_everything, Callback, Trainer
from torch.utils.data import Sampler, DataLoader
from torch.utils.data.dataset import T_co, Dataset
from tqdm import tqdm

from asme.core.modules.metrics_trait import MetricsTrait
from asme.data.datasets import ITEM_SEQ_ENTRY_NAME, SAMPLE_IDS, TARGET_ENTRY_NAME
from asme.core.metrics.container.metrics_container import MetricsContainer
from asme.core.metrics.metric import MetricStorageMode
from asme.core.init.context import Context
from asme.core.init.factories.metrics.metrics_container import MetricsContainerFactory
from asme.core.init.templating.search.configuration import SearchConfigurationTemplateProcessor
from asme.core.tokenization.utils.tokenization import remove_special_tokens
from asme.core.utils.run_utils import load_config, create_container, OBJECTIVE_METRIC_KEY, TRIAL_BASE_PATH, \
    load_and_restore_from_file_or_study, log_dataloader_example, load_hyperopt_config
from asme.core.utils import ioutils, logging
from asme.core.utils.ioutils import load_file_with_item_ids, determine_log_dir, save_config, save_finished_flag, \
    finished_flag_exists
from asme.core.writer.prediction.prediction_writer import build_prediction_writer
from asme.core.writer.results.results_writer import build_result_writer, check_file_format_supported

_ERROR_MESSAGE_LOAD_CHECKPOINT_FROM_FILE_OR_STUDY = "You have to specify at least the checkpoint file and config or" \
                                                    " the study name and study storage to infer the config and " \
                                                    "checkpoint path"

logger = logging.get_logger(__name__)
app = typer.Typer()


@app.command()
def train(config_file: Path = typer.Argument(..., help='the path to the config file', exists=True),
          do_resume: bool = typer.Option(False, help='flag iff the model should resume training from a checkpoint'),
          print_train_val_examples: bool = typer.Option(True, help='print examples of the training '
                                                                   'and evaluation dataset before starting training')
          ) -> None:
    config_file_path = Path(config_file)
    config = load_config(config_file_path)

    container = create_container(config)
    trainer = container.trainer().build()

    # save plain json config to the log dir/root dir of the trainer
    log_dir = determine_log_dir(trainer)
    save_config(config, log_dir)

    if do_resume:
        resume(log_dir, None)

    else:
        train_dataloader = container.train_dataloader()
        validation_dataloader = container.validation_dataloader()

        if print_train_val_examples:
            tokenizers = container.tokenizers()
            log_dataloader_example(train_dataloader, tokenizers, 'training')
            log_dataloader_example(validation_dataloader, tokenizers, 'validation')

        trainer.fit(container.module(),
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
    test_metrics_config = test_config.get_config(['module', 'metrics'])

    metrics_factory = MetricsContainerFactory()
    test_metrics_container = metrics_factory.build(test_metrics_config, Context())
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
        hyperopt_config = load_hyperopt_config(hyperopt_config_file, [SearchTemplateProcessor(OptunaParameterResolver(trial))])


        # infer model train directory from checkpoint output path set via `SearchConfigurationTemplateProcessor`.
        output_path = Path(config.get([TRAINER_CONFIG_KEY, CHECKPOINT_CONFIG_KEY, CHECKPOINT_CONFIG_DIR_PATH])).parent

        if not output_path.exists():
            output_path.mkdir(parents=True)

        shutil.copy2(hyperopt_config_file, output_path / "hyperopt_study.jsonnet")
        with (output_path / "hyperopt_trial.jsonnet").open("w") as hyperopt_config_file:
            json.dump(hyperopt_config.config, hyperopt_config_file, indent=2)

        patched_config = config.patch(hyperopt_config)

        #FIXME: adhoc fix to make output_path available in the configuration, remove when this is common in a configuration.
        patched_config.set_if_absent(["output_path"], str(output_path))
        return patched_config

    study = optuna.create_study(study_name=study_name,
                                storage=study_storage,
                                load_if_exists=True,
                                direction=study_direction)
    study.set_user_attr(OBJECTIVE_METRIC_KEY, objective_metric)

    for trial_run in range(1, num_trials+1):
        print(f"Running trial {trial_run} / {num_trials}")
        trial = study.ask()

        trial_study_direction = trial.study.direction
        objective_best = {
            StudyDirection.MINIMIZE: min,
            StudyDirection.MAXIMIZE: max
        }[trial_study_direction]

        # load config template, apply processors and write to `tmp_config_file`
        model_config = config_from_template(template_file, optimization_config_file, trial)

        class MetricsHistoryCallback(Callback):
            """
            Captures the reported metrics after every validation epoch.
            """

            def __init__(self):
                super().__init__()

                self.metric_history = []

            def on_validation_end(self, pl_trainer, pl_module):
                self.metric_history.append(pl_trainer.callback_metrics)

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
        save_config(model_config, log_dir)
        save_mlflow_id(trainer, log_dir)

        trainer.fit(
            module,
            train_dataloader=container.train_dataloader(),
            val_dataloaders=container.validation_dataloader()
        )
        save_finished_flag(log_dir)

        def _find_best_value(key: str, best: Callable[[List[float]], float] = min) -> float:
            values = [history_entry[key] for history_entry in metrics_tracker.metric_history]
            return best(values)

        study.tell(trial, _find_best_value(objective_metric, objective_best))


@app.command()
def predict(output_file: Path = typer.Argument(..., help='path where output is written'),
            num_predictions: int = typer.Option(default=20, help='number of predictions to export'),
            gpu: Optional[int] = typer.Option(default=0, help='number of gpus to use.'),
            selected_items_file: Optional[Path] = typer.Option(default=None,
                                                               help='only use the item ids for prediction'),
            checkpoint_file: Path = typer.Option(default=None, help='path to the checkpoint file'),
            config_file: Path = typer.Option(default=None, help='the path to the config file'),
            study_name: str = typer.Option(default=None, help='the study name of an existing study'),
            study_storage: str = typer.Option(default=None, help='the connection string for the study storage'),
            overwrite: Optional[bool] = typer.Option(default=False, help='overwrite output file if it exists.'),
            log_input: Optional[bool] = typer.Option(default=True, help='enable input logging.'),
            log_per_sample_metrics: Optional[bool] = typer.Option(default=True,
                                                                  help='enable logging of per-sample metrics.'),
            seed: Optional[int] = typer.Option(default=None, help='seed used eg for the sampled evaluation')
            ):
    """

    writes the predictions of model (restored from a checkpoint file) to a output file
    the checkpoint file can be provided as argument or is automatically inferred from the best trail of
    the provided study

    :param study_name: the name of the study
    :param study_storage: the storage uri of the study
    :param config_file: the config file used while training the model
    :param output_file: the path to write the output to
    :param num_predictions: number of predictions
    :param checkpoint_file: the checkpoint file of the model
    :param gpu: the number of gpus to use
    :param selected_items_file: the item that should only be considered
    :param overwrite: override the output file
    :param log_input: write the input sequence also to the file
    :param log_per_sample_metrics: if true also writes per sample metrics for the samples
    :param seed: the seed to use for this model (should not effect the predictions but the metrics if sampled)
    """

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

    if log_per_sample_metrics:
        metrics_container: MetricsContainer = module.metrics
        for metric in metrics_container.get_metrics():
            metric.set_metrics_storage_mode(MetricStorageMode.PER_SAMPLE)

    def _noop_filter(sample_predictions: np.ndarray):
        return sample_predictions

    filter_predictions = _noop_filter
    selected_items = None

    if selected_items_file is not None:
        selected_items = load_file_with_item_ids(selected_items_file)

        def _selected_items_filter(sample_predictions: np.ndarray):
            return sample_predictions[selected_items]

        filter_predictions = _selected_items_filter

    # open the file and build the writer
    with open(output_file, 'w') as result_file:
        output_writer = build_prediction_writer(result_file, log_input)

        # XXX: currently the predict method returns all batches at once, this is not RAM efficient
        # so we loop through the loader and use only one batch to call the predict method of pytorch lightning
        # replace as soon as this is fixed in pytorch lighting
        class FixedBatchSampler(Sampler):

            def __init__(self,
                         batch_start: int,
                         batch_size: int):
                super().__init__(None)
                self.batch_start = batch_start
                self.batch_size = batch_size

            def __iter__(self) -> Iterator[T_co]:
                return iter([range(self.batch_start, self.batch_start + self.batch_size)])

            def __len__(self):
                return 1

        item_tokenizer = container.tokenizer('item')

        def _extract_sample_metrics(module: MetricsTrait) -> List[Tuple[str, torch.Tensor]]:
            """
            Extracts the raw values of all metrics with per-sample-storage enabled.
            :param module: The module used for generating predictions.
            :return: A list of all metrics in the module's metric container with per-sample-storage enabled.
            """
            metrics_container = module.metrics
            metric_names_and_values = list(filter(lambda x: x[1]._storage_mode == MetricStorageMode.PER_SAMPLE,
                                                  zip(metrics_container.get_metric_names(),
                                                      metrics_container.get_metrics())))
            return list(map(lambda x: (x[0], x[1].raw_metric_values()), metric_names_and_values))

        def _create_batch_loader(dataset: Dataset,
                                 batch_sampler: Sampler,
                                 collate_fn,
                                 num_workers: int
                                 ) -> DataLoader:
            return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=num_workers)

        @contextmanager
        def _no_eval_step_end_call(module):
            """
            Wrap a call to trainer.test with this context manager to avoid the _eval_epoch_end code provided by the
            MetricsTrait to be executed before the wrapped code is executed. The hook is called afterwards.
            """
            _eval_epoch_end_hook = module._eval_epoch_end
            try:
                module._eval_epoch_end = lambda x: {}
                yield None
            finally:
                module._eval_epoch_end = _eval_epoch_end_hook
                module._eval_epoch_end(None)

        for index, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            sequences = batch[ITEM_SEQ_ENTRY_NAME]
            batch_size = sequences.size()[0]
            is_basket_recommendation = len(sequences.size()) == 3
            batch_start = index * batch_size

            # We need two loaders since we have to run both, predict & test on each batch
            batch_loader_predict = _create_batch_loader(test_loader.dataset,
                                                        batch_sampler=FixedBatchSampler(batch_start, batch_size),
                                                        collate_fn=test_loader.collate_fn,
                                                        num_workers=test_loader.num_workers)
            batch_loader_test = _create_batch_loader(test_loader.dataset,
                                                     batch_sampler=FixedBatchSampler(batch_start, batch_size),
                                                     collate_fn=test_loader.collate_fn,
                                                     num_workers=test_loader.num_workers)

            # Redirect prediction/test results to /dev/null to avoid spamming stdout
            with open(os.devnull, "w") as f, redirect_stdout(f):
                prediction_results = trainer.predict(module, dataloaders=batch_loader_predict)
                if log_per_sample_metrics:
                    # Prevent the reset-method of metrics to be called before extracting their values
                    with _no_eval_step_end_call(module):
                        # Run test in order to generate metrics (not generated by predict)
                        trainer.test(module, test_dataloaders=batch_loader_test)
                        metrics = _extract_sample_metrics(module)
                else:
                    metrics = []
            predictions = prediction_results[0]
            sample_ids = batch[SAMPLE_IDS]
            sequence_position_ids = None
            if 'pos' in batch:
                sequence_position_ids = batch['pos']
            targets = batch[TARGET_ENTRY_NAME]

            def _softmax(array: np.array) -> np.array:
                return np.exp(array) / sum(np.exp(array))

            def _generate_sample_id(sample_ids, sequence_position_ids, sample_index) -> str:
                sample_id = sample_ids[sample_index].item()
                if sequence_position_ids is None:
                    return sample_id

                return f'{sample_id}_{sequence_position_ids[sample_index].item()}'

            for i in range(predictions.shape[0]):
                prediction = filter_predictions(predictions[i])

                scores = _softmax(prediction)

                item_indices = scores.argsort()[::-1][:num_predictions]

                item_ids = item_indices.tolist()

                # when we only want the predictions of selected items
                # the indices are not the item ids anymore, so we have to update them here
                if selected_items is not None:
                    selected_item_ids = [selected_items[i] for i in item_ids]
                    item_ids = selected_item_ids

                tokens = item_tokenizer.convert_ids_to_tokens(item_ids)
                scores.sort()
                scores = scores[::-1].tolist()[:num_predictions]

                sample_id = _generate_sample_id(sample_ids, sequence_position_ids, i)
                true_target = targets[i]
                if is_basket_recommendation:
                    true_target = remove_special_tokens(true_target.tolist(), item_tokenizer)
                else:
                    true_target = true_target.item()
                true_target = item_tokenizer.convert_ids_to_tokens(true_target)
                metric_name_and_values = [(name, value[0][i].item()) for name, value in metrics]
                sequence = None
                if log_input:
                    sequence = sequences[i].tolist()

                    # remove padding tokens
                    sequence = remove_special_tokens(sequence, item_tokenizer)
                    sequence = item_tokenizer.convert_ids_to_tokens(sequence)

                output_writer.write_values(f'{sample_id}', tokens, scores, true_target, metric_name_and_values,
                                           sequence)


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

    eval_results = trainer.test(module, test_dataloaders=test_loader, verbose=not write_results_to_file)

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
    config_file = log_dir / ioutils.PROCESSED_CONFIG_NAME
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

    trainer.fit(module, train_dataloader=train_loader, val_dataloaders=validation_loader)

    # Save finished flag
    save_finished_flag(log_dir)


if __name__ == "__main__":
    app()
