from pathlib import Path
from typing import Any, Dict, Union, Iterable, Optional

from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger, MLFlowLogger, WandbLogger

TRAINER_INIT_KEYS = ['logger',
                     'checkpoint_callback',
                     'callbacks',
                     'default_root_dir',
                     'gradient_clip_val',
                     'process_position',
                     'num_nodes',
                     'num_processes',
                     'gpus',
                     'auto_select_gpus',
                     'tpu_cores',
                     'log_gpu_memory',
                     'progress_bar_refresh_rate',
                     'overfit_batches',
                     'track_grad_norm',
                     'check_val_every_n_epoch',
                     'fast_dev_run',
                     'accumulate_grad_batches',
                     'max_epochs',
                     'min_epochs',
                     'max_steps',
                     'min_steps',
                     'limit_train_batches',
                     'limit_val_batches',
                     'limit_test_batches',
                     'val_check_interval',
                     'flush_logs_every_n_steps',
                     'log_every_n_steps',
                     'accelerator',
                     'sync_batchnorm',
                     'precision',
                     'weights_summary',
                     'weights_save_path',
                     'num_sanity_val_steps',
                     'truncated_bptt_steps',
                     'resume_from_checkpoint',
                     'profiler',
                     'benchmark',
                     'deterministic',
                     'reload_dataloaders_every_epoch',
                     'auto_lr_find',
                     'replace_sampler_ddp',
                     'terminate_on_nan',
                     'auto_scale_batch_size',
                     'prepare_data_per_node',
                     'plugins',
                     'amp_backend',
                     'amp_level',
                     'distributed_backend',
                     'automatic_optimization',
                     'move_metrics_to_cpu',
                     'enable_pl_optimizer']


class TrainerBuilder:

    def __init__(self, trainer_parameters: Dict[str, Any] = None, strict=False, **kwargs):
        self.kwargs = kwargs
        self.callbacks = []
        self.loggers = []
        self.strict = strict

        if trainer_parameters is not None:
            self.load_dict(trainer_parameters)

    def from_checkpoint(self, checkpoint_path: Union[str, Path]):
        return self.set("resume_from_checkpoint", checkpoint_path)

    def get(self, key: str) -> Optional[Any]:
        if key in self.kwargs:
            return self.kwargs[key]

        return None

    def set(self, key: str, value: Any):
        if key not in TRAINER_INIT_KEYS:
            if self.strict:
                raise KeyError(f"Invalid key '{key}' for Trainer constructor.")
        else:
            self.kwargs[key] = value

        return self

    def load_dict(self, args: Dict[str, Any]):
        for key, value in args.items():
            self.set(key, value)

        return self

    def add_callback(self, callbacks: Union[Callback, Iterable[Callback]]):
        self.callbacks += [callbacks] if not isinstance(callbacks, Iterable) else callbacks
        return self

    def add_logger(self, loggers: Union[LightningLoggerBase, Iterable[LightningLoggerBase]]):
        self.loggers += [loggers] if not isinstance(loggers, Iterable) else loggers
        return self

    def add_checkpoint_callback(self, parameters: Dict[str, Any]):
        if "dirpath" not in parameters and "default_root_dir" in self.kwargs:
            parameters["dirpath"] = Path(self.kwargs["default_root_dir"]) / "checkpoints"
        if "filename" not in parameters:
            monitored_metric = parameters["monitor"]
            parameters['filename'] = f"{monitored_metric}"+"{epoch}"
        checkpoint = ModelCheckpoint(**parameters)
        return self.add_callback(checkpoint)

    def build(self) -> Trainer:
        # This is necessary since some of the keys occurring in the config are filled with actual objects by the builder
        sanitized_args = self.kwargs.copy()
        if "logger" in sanitized_args:
            logger_config = sanitized_args.pop("logger")
            if len(self.loggers) == 0:
                self.add_logger(LoggerBuilder(parameters=logger_config).build())

        # Build the actual trainer object from the parameters we got and the callbacks/loggers we constructed
        return Trainer(**sanitized_args, callbacks=self.callbacks, logger=self.loggers)


def _build_tensorboard_logger(parameters: Dict[str, Any]) -> LightningLoggerBase:
    return TensorBoardLogger(
        save_dir=parameters.get("log_dir"),
        name=parameters.get('experiment_name')
    )


def _build_mlflow_logger(parameters: Dict[str, Any]) -> LightningLoggerBase:
    return MLFlowLogger(
        experiment_name=parameters["experiment_name"],
        tracking_uri=parameters["tracking_uri"]
    )


def _build_wandb_logger(parameters: Dict[str, Any]) -> LightningLoggerBase:
    return WandbLogger(
        project=parameters['project'],
        log_model=parameters.get('log_model', False)
    )


LOGGER_REGISTRY = {
    'tensorboard': _build_tensorboard_logger,
    'mlflow': _build_mlflow_logger,
    'wandb': _build_wandb_logger
}


class LoggerBuilder:

    def __init__(self, name: str = None, parameters: Dict[str, Any] = None):
        self.parameters = parameters if parameters is not None else {}
        self.type = name

    def load_dict(self, parameters: Dict[str, Any]):
        for key, value in parameters:
            self.set(key, value)

        return self

    def set(self, key: str, value: Any):
        self.parameters[key] = value

        return self

    def build(self) -> LightningLoggerBase:
        if "type" in self.parameters:
            self.type = self.parameters.pop("type")
        if self.type is None:
            raise RuntimeError("Parameter 'type' has to be included in logger configuration")
        return LOGGER_REGISTRY[self.type.lower()](self.parameters)


CALLBACK_REGISTRY = {

}


class CallbackBuilder:

    def __init__(self, name: str = None, parameters: Dict[str, Any] = None):
        self.parameters = parameters if parameters is not None else {}
        self.type = name

    def load_dict(self, parameters: Dict[str, Any]):
        for key, value in parameters:
            self.set(key, value)

        return self

    def set(self, key: str, value: Any):
        self.parameters[key] = value

        return self

    def build(self) -> Callback:
        if "type" in self.parameters:
            self.type = self.parameters.pop("type")
        if self.type is None:
            raise RuntimeError("Parameter 'type' has to be set in callback configuration")
        return CALLBACK_REGISTRY[self.type.lower()](self.parameters)
