import os, errno
from pathlib import Path
from typing import Union, Optional, Dict, Any

import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer


class BestModelWritingModelCheckpoint(ModelCheckpoint):
    """
        Tracks best checkpoints by decorating the `ModelCheckpoint` class.

        Calls the to_yaml method after each epoch and creates a symlink `best.ckpt` pointing to the best model.
        Additionally, validation metrics for each checkpoint are stored in `<output_base_path>/<output_filename>`.
        If output_base_path can not be found in the configuration, the dirpath of model_checkpoint is used as the default.
        Same goes for output_filename where the default value is "best_k_models.yaml".
    """

    def __init__(self,
                 model_checkpoint: ModelCheckpoint,
                 output_base_path: str,
                 output_filename: str,
                 symlink_name: str = 'best.ckpt'
                 ):
        super().__init__()
        self.target_object = model_checkpoint
        if output_base_path is not None:
            self.output_base_path = output_base_path
        else:
            self.output_base_path = self.target_object.dirpath
        if output_filename is not None:
            self.output_filename = output_filename
        else:
            self.output_filename = "best_k_models.yaml"

        self.symlink_name = symlink_name

    def on_train_end(self, trainer, pl_module: LightningModule):
        self.target_object.on_train_end(trainer, pl_module)
        self.target_object.to_yaml(os.path.join(self.output_base_path, self.output_filename))
        symlink_path = os.path.join(self.output_base_path, self.symlink_name)
        # here we only link relative paths, to prevent wrong links when
        # the result path is mounted into a VM, container …
        best_checkpoint_path = Path(self.target_object.best_model_path).name
        try:
            os.symlink(best_checkpoint_path, symlink_path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                os.remove(symlink_path)
                os.symlink(best_checkpoint_path, symlink_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        self.target_object.on_pretrain_routine_start(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        self.target_object.on_validation_end(trainer, pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        return self.target_object.on_save_checkpoint(trainer, pl_module, checkpoint)

    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any], **kwargs):
        self.target_object.on_load_checkpoint(checkpointed_state, kwargs)

    def save_checkpoint(self, trainer: Trainer, unused: Optional[LightningModule] = None):
        self.target_object.save_checkpoint(trainer, unused)

    def _del_model(self, trainer: Trainer, filepath: str):
        return self.target_object._del_model(trainer, filepath)

    def _save_model(self, trainer: Trainer, filepath: str):
        return self.target_object._save_model(trainer, filepath)

    def check_monitor_top_k(self, trainer, current: Optional[torch.Tensor] = None) -> bool:
        return self.target_object.check_monitor_top_k(current)

    @classmethod
    def _format_checkpoint_name(cls,
                                filename: Optional[str],
                                metrics: Dict,
                                prefix: str = "",
                                auto_insert_metric_name: bool = True,
                                ) -> str:
        return super()._format_checkpoint_name(filename, metrics, prefix, auto_insert_metric_name)

    def format_checkpoint_name(self, metrics: Dict, ver: Optional[int] = None) -> str:
        return self.target_object.format_checkpoint_name(metrics, ver)

    def _validate_monitor_key(self, trainer):
        return self.target_object._validate_monitor_key(trainer)

    def _get_metric_interpolated_filepath_name(self, monitor_candidates: Dict, trainer: "pl.Trainer", del_filepath: Optional[str] = None) -> str:
        return self.target_object._get_metric_interpolated_filepath_name(monitor_candidates, trainer, del_filepath)

    def _monitor_candidates(self, trainer: "pl.Trainer", epoch: int, step: int):
        return self.target_object._monitor_candidates(trainer)

    def _save_last_checkpoint(self, trainer: "pl.Trainer", monitor_candidates: Dict):
        self.target_object._save_last_checkpoint(trainer, monitor_candidates)

    def _save_top_k_checkpoint(self, trainer: "pl.Trainer", monitor_candidates: Dict):
        self.target_object._save_top_k_checkpoint(trainer, monitor_candidates)

    def _is_valid_monitor_key(self, metrics):
        return self.target_object._is_valid_monitor_key(metrics)

    def _update_best_and_save(self, current: torch.Tensor, trainer: "pl.Trainer", monitor_candidates: Dict):
        self.target_object._update_best_and_save(current, trainer, monitor_candidates)

    def to_yaml(self, filepath: Optional[Union[str, Path]] = None):
        self.target_object.to_yaml(filepath)

    def file_exists(self, filepath: Union[str, Path], trainer) -> bool:
        return self.target_object.file_exists(filepath, trainer)
