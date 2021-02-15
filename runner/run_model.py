import logging
import logging.handlers
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Any, Optional, List, Callable

import pytorch_lightning as pl

import optuna
import typer
from dependency_injector import containers, providers
from optuna.structs import StudyDirection
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.utilities import cloud_io

from runner.util.builder import TrainerBuilder, LoggerBuilder, CallbackBuilder
from runner.util.containers import BERT4RecContainer, CaserContainer, SASRecContainer, NarmContainer, RNNContainer,\
    DreamContainer
from search.processor import ConfigTemplateProcessor
from search.resolver import OptunaParameterResolver

app = typer.Typer()


# TODO: introduce a subclass for all container configurations?
def build_container(model_id: str, config_file: str) -> containers.DeclarativeContainer:
    container = {
        'bert4rec': BERT4RecContainer(),
        'sasrec': SASRecContainer(),
        'caser': CaserContainer(),
        "narm": NarmContainer(),
        "rnn": RNNContainer(),
        'dream': DreamContainer()
    }[model_id]
    container.config.from_yaml(config_file)
    return container


# FIXME: progress bar is not logged :(
def _config_logging(config: Dict[str, Any]
                    ) -> None:
    logger = logging.getLogger("lightning")
    handler = logging.handlers.RotatingFileHandler(
        Path(config['trainer']['default_root_dir']) / 'run.log', maxBytes=(1048576 * 5), backupCount=7
    )
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _get_base_trainer_builder(config) -> TrainerBuilder:
    trainer_builder = TrainerBuilder(config.trainer())
    trainer_builder = trainer_builder.add_checkpoint_callback(config.trainer.checkpoint())
    trainer_builder = trainer_builder.add_logger(LoggerBuilder(parameters=config.trainer.logger()).build())
    return trainer_builder

def _build_trainer(config: providers.Configuration,
                   callbacks: List[Callback] = None
                   ) -> pl.Trainer:
    trainer_builder = TrainerBuilder(config())
    trainer_builder = trainer_builder.add_checkpoint_callback(config.checkpoint())
    trainer_builder = trainer_builder.add_logger(LoggerBuilder(parameters=config.logger()).build())
    if callbacks:
        for callback_logger in callbacks:
            trainer_builder = trainer_builder.add_callback(callback_logger)
    trainer = trainer_builder.build()
    return trainer


@app.command()
def train(model: str = typer.Argument(..., help="the model to run"),
          config_file: str = typer.Argument(..., help='the path to the config file'),
          do_train: bool = typer.Option(True, help='flag iff the model should be trained'),
          do_test: bool = typer.Option(False, help='flag iff the model should be tested (after training)')
          ) -> None:
    # XXX: because the dependency injector does not provide a error message when the config file does not exists,
    # we manually check if the config file exists
    if not os.path.isfile(config_file):
        print(f"the config file cannot be found. Please check the path '{config_file}'!")
        exit(-1)

    container = build_container(model, config_file)
    module = container.module()

    config = container.config
    trainer_builder = _get_base_trainer_builder(config)
    trainer = trainer_builder.build()

    if do_train:
        trainer.fit(module, train_dataloader=container.train_loader(), val_dataloaders=container.validation_loader())

    if do_test:
        if not do_train:
            print(f"The model has to be trained before it can be tested!")
            exit(-1)
        trainer.test(test_dataloaders=container.test_loader())


@app.command()
def search(model: str = typer.Argument(..., help="the model to run"),
           template_file: Path = typer.Argument(..., help='the path to the config file'),
           study_name: str = typer.Argument(..., help='the study name of an existing optuna study'),
           study_storage: str = typer.Argument(..., help='the connection string for the study storage'),
           objective_metric: str = typer.Argument(..., help='the name of the metric to watch during the study'
                                                            '(e.g. recall_at_5).'),
           num_trails: int = typer.Option(default=20, help='the number of trails to execute')
          ) -> None:
    # XXX: because the dependency injector does not provide a error message when the config file does not exists,
    # we manually check if the config file exists
    if not os.path.isfile(template_file):
        print(f"the config file cannot be found. Please check the path '{template_file}'!")
        exit(-1)

    def config_from_template(template_file: Path,
                             config_file_handle,
                             trial):
        import yaml
        resolver = OptunaParameterResolver(trial)
        processor = ConfigTemplateProcessor(resolver)

        with template_file.open("r") as f:
            template = yaml.load(f)
            resolved_config = processor.process(template)

            yaml.dump(resolved_config, config_file_handle)
            config_file_handle.flush()

    def objective(trial: optuna.Trial):
        # get the direction to get if we must extract the max or the min value of the metric
        study_direction = trial.study.direction
        objective_best = {StudyDirection.MINIMIZE: min, StudyDirection.MAXIMIZE: max}[study_direction]

        with NamedTemporaryFile(mode='wt') as tmp_config_file:
            config_from_template(template_file, tmp_config_file, trial)

            class MetricsHistoryCallback(Callback):

                def __init__(self):
                    super().__init__()

                    self.metric_history = []

                def on_validation_end(self, trainer, pl_module):
                    self.metric_history.append(trainer.callback_metrics)

            metrics_tracker = MetricsHistoryCallback()

            container = build_container(model, tmp_config_file.name)

            module = container.module()

            config = container.config
            trainer = _build_trainer(config.trainer, callbacks=[metrics_tracker])
            trainer.fit(module, train_dataloader=container.train_loader(), val_dataloaders=container.validation_loader())

            def _find_best_value(key: str, best: Callable[[List[float]], float] = min) -> float:
                values = [history_entry[key] for history_entry in metrics_tracker.metric_history]
                return best(values)

            return _find_best_value(objective_metric, objective_best)

    study = optuna.load_study(study_name=study_name, storage=study_storage)
    study.optimize(objective, n_trials=num_trails)


@app.command()
def predict(model: str = typer.Argument(..., help="the model to run"),
            config_file: str = typer.Argument(..., help='the path to the config file'),
            checkpoint_file: str = typer.Argument(..., help='path to the checkpoint file'),
            output_file: Path = typer.Argument(..., help='path where output is written'),
            gpu: Optional[int] = typer.Option(default=0, help='number of gpus to use.'),
            overwrite: Optional[bool] = typer.Option(default=False, help='overwrite output file if it exists.'),
            log_input: Optional[bool] = typer.Option(default=False, help='enable input logging.'),
            strip_pad_token: Optional[bool] = typer.Option(default=True, help='strip pad token, if input is logged.')
            ):
    if not overwrite and output_file.exists():
        print(f"${output_file} already exists. If you want to overwrite it, use `--overwrite`.")
        exit(-1)

    container = build_container(model, config_file)
    module = container.module()

    # FIXME: try to use load_from_checkpoint later
    # load checkpoint <- we don't use the PL function load_from_checkpoint because it does
    # not work with our module class system
    ckpt = cloud_io.load(checkpoint_file)

    # acquire state_dict
    state_dict = ckpt["state_dict"]

    # load parameters and freeze the model
    module.load_state_dict(state_dict)
    module.freeze()

    test_loader = container.test_loader()

    callback_params = {
        "output_file_path": output_file,
        "log_input": log_input,
        "tokenizer": container.tokenizer(),
        "strip_padding_tokens": strip_pad_token
    }
    config = container.config
    trainer_builder = TrainerBuilder(config.trainer())
    trainer_builder = trainer_builder.add_callback(CallbackBuilder("prediction_logger", callback_params).build())
    trainer_builder = trainer_builder.set("gpus", gpu)
    trainer = trainer_builder.build()

    trainer.test(module, test_dataloaders=test_loader)


# FIXME: (AD) metrics are not calculated correctly
# FIXME: (AD) need to write output to file, but first need to resolve Exception caused by trainer test loop :-/
@app.command()
def evaluate(model: str = typer.Argument(..., help="the model to run"),
             config_file: str = typer.Argument(..., help='the path to the config file'),
             checkpoint_file: str = typer.Argument(..., help='path to the checkpoint file'),
             output_file: Path = typer.Argument(..., help='path where output is written'),
             gpu: Optional[int] = typer.Option(default=0, help='number of gpus to use.'),
             overwrite: Optional[bool] = typer.Option(default=False, help='overwrite output file if it exists.'),
             seed: Optional[int] = typer.Option(default=42, help='seed for rng')
             ):
    if not overwrite and output_file.exists():
        print(f"${output_file} already exists. If you want to overwrite it, use `--overwrite`.")
        exit(-1)

    seed_everything(seed)

    container = build_container(model, config_file)
    module = container.module()

    # FIXME: try to use load_from_checkpoint later
    # load checkpoint <- we don't use the PL function load_from_checkpoint because it does
    # not work with our module class system
    ckpt = cloud_io.load(checkpoint_file)

    # acquire state_dict
    state_dict = ckpt["state_dict"]

    # load parameters and freeze the model
    module.load_state_dict(state_dict)
    module.freeze()

    test_loader = container.test_loader()

    trainer_builder = TrainerBuilder(gpus=gpu)

    trainer = trainer_builder.build()
    trainer.test(module, test_dataloaders=test_loader)


@app.command()
def resume(model: str = typer.Argument(..., help="the model to run."),
           config_file: str = typer.Argument(..., help='the path to the config file'),
           checkpoint_file: str = typer.Argument(..., help="path to the checkpoint file.")):
    container = build_container(model, config_file)
    module = container.module()

    config = container.config
    trainer = _get_base_trainer_builder(config).from_checkpoint(checkpoint_file).build()

    train_loader = container.train_loader()
    validation_loader = container.validation_loader()

    trainer.fit(module, train_dataloader=train_loader, val_dataloaders=validation_loader)


if __name__ == "__main__":
    app()
