from typing import Dict, Optional

import torch
from pytorch_lightning.utilities import rank_zero_warn
from torch.nn.parameter import Parameter

from asme.core.utils.hyperparameter_utils import save_hyperparameters
from asme.core.utils.inject import InjectTokenizer, inject
from asme.data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from asme.core.metrics.container.metrics_container import MetricsContainer
from asme.core.modules.metrics_trait import MetricsTrait
from pytorch_lightning import core as pl

from asme.core.modules.util.module_util import build_eval_step_return_dict, get_padding_mask
from asme.core.modules.util.noop_optimizer import NoopOptimizer
from asme.core.tokenization.tokenizer import Tokenizer


class GenrePopModule(MetricsTrait, pl.LightningModule):
    """
    module that provides a baseline, that returns the most popular items in the dataset for recommendation
    """

    @inject(item_tokenizer=InjectTokenizer("item"))
    @inject(attr_tokenizer=InjectTokenizer("attr"))
    @save_hyperparameters
    def __init__(self,
                 item_tokenizer: Tokenizer,
                 attr_tokenizer: Tokenizer,
                 metrics: MetricsContainer
                 ):
        super().__init__()
        self.item_tokenizer = item_tokenizer
        self.attr_tokenizer = attr_tokenizer
        self.item_vocab_size = len(item_tokenizer)
        self.attr_vocab_size = len(attr_tokenizer)
        self.metrics = metrics

        # we artificially promote this Tensor to a parameter to make PL save it in the model checkpoints
        self.item_frequencies = Parameter(torch.ones(self.attr_vocab_size, self.item_vocab_size,
                                                      device=self.device), requires_grad=False)

        self.save_hyperparameters(self.hyperparameters)

    def on_train_start(self) -> None:
        if self.trainer.max_epochs > 1:
            rank_zero_warn(
                f"When training the POP baseline, "
                f"'trainer.max_epochs' should be set to 1 (but is {self.trainer.max_epochs}).")

    def get_metrics(self) -> MetricsContainer:
        return self.metrics

    def forward(self,
                batch: Dict[str, torch.Tensor],
                batch_idx: Optional[int] = None
                ) -> torch.Tensor:
        batch_size = batch[ITEM_SEQ_ENTRY_NAME].shape[0]

        item_input = batch[ITEM_SEQ_ENTRY_NAME]
        attr_input = batch["attr.target"]

        target_attribute_item_frequencies = self.item_frequencies[attr_input]
        # We rank the items in order of frequency
        predictions = target_attribute_item_frequencies / target_attribute_item_frequencies.sum(0)
        return predictions

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int
                      ):
        input_seq = torch.flatten(batch[ITEM_SEQ_ENTRY_NAME])
        input_seq_attributes = torch.flatten(batch["attr"])
        mask = get_padding_mask(input_seq, self.item_tokenizer)
        masked_input = input_seq * mask
        masked_attributes = input_seq_attributes * mask
        masked_input = masked_input[masked_input > 1]
        masked_attributes = masked_attributes[masked_attributes > 1]

        counts = torch.stack([torch.bincount(masked_input[masked_attributes == x], minlength=self.item_vocab_size) for x in
                  range(self.attr_vocab_size)])

        self.item_frequencies += counts
        return {
            "loss": torch.tensor(0., device=self.device)
        }

    def eval_step(self,
                  batch: Dict[str, torch.Tensor],
                  batch_idx: int
                  ) -> Dict[str, torch.Tensor]:
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        targets = batch[TARGET_ENTRY_NAME]
        prediction = self.forward(batch)

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
