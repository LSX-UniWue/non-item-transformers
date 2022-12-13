from datetime import timedelta
from pathlib import Path
from typing import Optional, Dict, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


class BestModelWritingModelCheckpoint(ModelCheckpoint):
    """
        Tracks best checkpoints by decorating the `ModelCheckpoint` class.

        Calls the to_yaml method after each epoch and creates a symlink `best.ckpt` pointing to the best model.
        Additionally, validation metrics for each checkpoint are stored in `<output_base_path>/<output_filename>`.
        If output_base_path can not be found in the configuration, the dirpath of model_checkpoint is used as the default.
        Same goes for output_filename where the default value is "best_k_models.yaml".
    """
    def __init__(self,
                 dirpath: Optional[Union[str, Path]] = None,
                 filename: Optional[str] = None,
                 monitor: Optional[str] = None,
                 verbose: bool = False,
                 save_last: Optional[bool] = None,
                 save_top_k: int = 1,
                 save_weights_only: bool = False,
                 mode: str = "min",
                 auto_insert_metric_name: bool = True,
                 every_n_train_steps: Optional[int] = None,
                 train_time_interval: Optional[timedelta] = None,
                 every_n_epochs: Optional[int] = None,
                 save_on_train_epoch_end: Optional[bool] = None,
                 skip_best: bool = False
                 ):
        self.skip_best = skip_best
        super().__init__(dirpath,
                         filename,
                         monitor,
                         verbose,
                         save_last,
                         save_top_k,
                         save_weights_only,
                         mode,
                         auto_insert_metric_name,
                         every_n_train_steps,
                         train_time_interval,
                         every_n_epochs,
                         save_on_train_epoch_end)

        self.last_best_model_path = ""

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_end(trainer, pl_module)

        if not self._should_skip_saving_best_checkpoint(trainer) and self._save_on_train_epoch_end:
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                self._save_best_model_checkpoint_symlink()
                self.last_best_model_path = self.best_model_path

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_validation_end(trainer, pl_module)

        if not self._should_skip_saving_best_checkpoint(trainer) and not self._save_on_train_epoch_end:
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                self._save_best_model_checkpoint_symlink()
                self.last_best_model_path = self.best_model_path

    def _should_skip_saving_best_checkpoint(self, trainer: "pl.Trainer") -> bool:
        from pytorch_lightning.trainer.states import TrainerFn
        return (
                trainer.fast_dev_run  # disable checkpointing with fast_dev_run
                or trainer.state.fn != TrainerFn.FITTING  # don't save anything during non-fit
                or trainer.sanity_checking  # don't save anything during sanity check
                or self.last_best_model_path == self.best_model_path
                or self.skip_best
        )

    def _save_best_model_checkpoint_symlink(self):
        symlink_path = Path(self.dirpath) / "best.ckpt"
        # here we only link relative paths, to prevent wrong links when
        # the result path is mounted into a VM, container …
        best_checkpoint_path = Path(self.best_model_path).name

        symlink_path.unlink(missing_ok=True)
        symlink_path.symlink_to(best_checkpoint_path)

    @classmethod
    def _format_checkpoint_name(cls,
                                filename: Optional[str],
                                metrics: Dict,
                                prefix: str = "",
                                auto_insert_metric_name: bool = True,
                                ) -> str:
        return super()._format_checkpoint_name(filename, metrics, prefix, auto_insert_metric_name)

