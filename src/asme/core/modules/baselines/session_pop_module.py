from typing import Dict

import torch
from pytorch_lightning.utilities import rank_zero_warn

from asme.core.utils.hyperparameter_utils import save_hyperparameters
from asme.data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from asme.core.metrics.container.metrics_container import MetricsContainer
from asme.core.modules.metrics_trait import MetricsTrait
from pytorch_lightning import core as pl

from asme.core.modules.util.module_util import build_eval_step_return_dict, get_padding_mask
from asme.core.modules.util.noop_optimizer import NoopOptimizer
from asme.core.tokenization.tokenizer import Tokenizer


class SessionPopModule(MetricsTrait, pl.LightningModule):

    @save_hyperparameters
    def __init__(self,
                 item_tokenizer: Tokenizer,
                 metrics: MetricsContainer
                 ):
        super().__init__()
        self.item_vocab_size = len(item_tokenizer)
        self.tokenizer = item_tokenizer
        self.metrics = metrics

        self.save_hyperparameters(self.hyperparameters)

    def on_train_start(self) -> None:
        if self.trainer.max_epochs > 1:
            rank_zero_warn(
                f"When training the SessionPOP baseline, "
                f"'trainer.max_epochs' should be set to 1 (but is {self.trainer.max_epochs}).")

    def get_metrics(self) -> MetricsContainer:
        return self.metrics

    def forward(self,
                input_seq: torch.Tensor
                ) -> torch.Tensor:
        batch_size = input_seq.shape[0]
        padding_mask = get_padding_mask(input_seq, self.tokenizer)
        masked = input_seq * padding_mask
        # We simply predict 0's for all but the most frequently seen item in the session
        predictions = torch.zeros((batch_size, self.item_vocab_size), device=self.device)
        # TODO: Find a way to do this without loops
        for i in range(batch_size):
            session = masked[i]
            # Find the most frequent item per session
            most_frequent, _ = torch.mode(session[session > 0])
            predictions[i, most_frequent] = 1
        return predictions

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx
                      ) -> Dict[str, float]:
        # since we simply predict the most frequent item of each session, we do not need to train at all
        return {
            'loss': 0.0
        }

    def eval_step(self,
                  batch: Dict[str, torch.Tensor],
                  batch_idx: int
                  ) -> Dict[str, torch.Tensor]:
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        targets = batch[TARGET_ENTRY_NAME]
        prediction = self.forward(input_seq)

        return build_eval_step_return_dict(input_seq, prediction, targets)

    def validation_step(self,
                        batch: Dict[str, torch.Tensor],
                        batch_idx: int
                        ) -> Dict[str, torch.Tensor]:
        return self.eval_step(batch, batch_idx)

    def test_step(self,
                  batch: Dict[str, torch.Tensor],
                  batch_idx: int):
        return self.eval_step(batch, batch_idx)

    # Do nothing on backward since we only count occurrences
    def backward(self,
                 loss: torch.Tensor,
                 optimizer: torch.optim.Optimizer,
                 optimizer_idx: int,
                 *args,
                 **kwargs
                 ) -> None:
        pass

    def configure_optimizers(self):
        return NoopOptimizer()
