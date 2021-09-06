from typing import Dict

import torch
from pytorch_lightning.utilities import rank_zero_warn
from torch.nn.parameter import Parameter

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from asme.metrics.container.metrics_container import MetricsContainer
from asme.modules.metrics_trait import MetricsTrait
from pytorch_lightning import core as pl

from asme.modules.util.module_util import build_eval_step_return_dict, get_padding_mask
from asme.modules.util.noop_optimizer import NoopOptimizer
from asme.tokenization.tokenizer import Tokenizer


class PopModule(MetricsTrait, pl.LightningModule):

    """
    module that provides a baseline, that returns the most popular items in the dataset for recommendation
    """

    def __init__(self,
                 item_tokenizer: Tokenizer,
                 metrics: MetricsContainer
                 ):
        super().__init__()
        self.item_tokenizer = item_tokenizer
        self.item_vocab_size = len(item_tokenizer)
        self.metrics = metrics

        # we artificially promote this Tensor to a parameter to make PL save it in the model checkpoints
        self.item_frequencies = Parameter(torch.zeros(self.item_vocab_size, device=self.device), requires_grad=False)

    def on_train_start(self) -> None:
        if self.trainer.max_epochs > 1:
            rank_zero_warn(
                f"When training the POP baseline, "
                f"'trainer.max_epochs' should be set to 1 (but is {self.trainer.max_epochs}).")

    def get_metrics(self) -> MetricsContainer:
        return self.metrics

    def forward(self,
                input_seq: torch.Tensor
                ) -> torch.Tensor:
        batch_size = input_seq.shape[0]
        # We rank the items in order of frequency
        predictions = torch.unsqueeze(self.item_frequencies / self.item_frequencies.sum(), dim=0).repeat(batch_size, 1)
        return predictions

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int
                      ):
        input_seq = torch.flatten(batch[ITEM_SEQ_ENTRY_NAME])
        mask = get_padding_mask(input_seq, self.item_tokenizer)
        masked = input_seq * mask
        masked = masked[masked > 0]
        self.item_frequencies += torch.bincount(masked, minlength=self.item_vocab_size)

        return {
            "loss": torch.tensor(0., device=self.device)
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
                        batch: Dict[str, torch.tensor],
                        batch_idx: int
                        ) -> Dict[str, torch.Tensor]:
        return self.eval_step(batch, batch_idx)

    def test_step(self,
                  batch: Dict[str, torch.Tensor],
                  batch_idx: int
                  ) -> Dict[str, torch.Tensor]:
        return self.eval_step(batch, batch_idx)

    # Do nothing on backward since we only count occurrences
    def backward(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, optimizer_idx: int, *args,
                 **kwargs) -> None:
        pass

    def configure_optimizers(self):
        return NoopOptimizer()
