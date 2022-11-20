from abc import abstractmethod
from typing import Union, Dict, Optional
import inspect

import torch

from asme.core.tokenization.tokenizer import Tokenizer
from asme.core.utils.hyperparameter_utils import save_hyperparameters
from asme.data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME, POSITIVE_SAMPLES_ENTRY_NAME, \
    NEGATIVE_SAMPLES_ENTRY_NAME
from asme.core.metrics.container.metrics_container import MetricsContainer
from asme.core.modules import LOG_KEY_VALIDATION_LOSS, LOG_KEY_TRAINING_LOSS
from asme.core.modules.metrics_trait import MetricsTrait
from asme.core.modules.util.module_util import get_padding_mask, build_eval_step_return_dict, get_additional_meta_data
from asme.core.modules.next_item_prediction_training_module import BaseNextItemPredictionTrainingModule
from asme.core.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.core.utils.inject import InjectTokenizer, inject
from asme.core.losses.losses import CrossEntropyLoss, SequenceRecommenderLoss, SingleTargetCrossEntropyLoss

class UserNextItemPredictionTrainingModule(BaseNextItemPredictionTrainingModule):


    @inject(
        item_tokenizer=InjectTokenizer("item")
    )
    @save_hyperparameters
    def __init__(self,
                 model: SequenceRecommenderModel,
                 item_tokenizer: Tokenizer,
                 metrics: MetricsContainer,
                 learning_rate: float = 0.001,
                 beta_1: float = 0.99,
                 beta_2: float = 0.998,
                 weight_decay: float = 0,
                 loss_function: Optional[SequenceRecommenderLoss] = None,
                 first_item: bool = False
                 ):

        super().__init__(model=model,metrics=metrics,learning_rate=learning_rate,beta_1=beta_1,beta_2=beta_2,
                         weight_decay=weight_decay,loss_function=loss_function)
        self.user_key_len = len(model.optional_metadata_keys())
        self.first_item = first_item


    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int
                      ) -> Optional[Union[torch.Tensor, Dict[str, Union[torch.Tensor, float]]]]:
        """
        Performs a validation step on a batch of sequences and returns the overall loss.

        `batch` must be a dictionary containing the following entries:
            * `ITEM_SEQ_ENTRY_NAME`: a tensor of size (N, S),
            * `TARGET_ENTRY_NAME`: a tensor of size (N) with the target items,

        :param batch: the batch
        :param batch_idx: the batch number.
        :return: A dictionary with entries according to `build_eval_step_return_dict`.

        where N is the batch size and S the max sequence length.
        """

        logits = self(batch, batch_idx)
        target = batch[TARGET_ENTRY_NAME]

        if self.first_item:
            target_logits = logits
        else:
            target_logits = self._extract_target_item_logits(logits)

        loss = self._calc_loss(target_logits, target)
        self.log(LOG_KEY_TRAINING_LOSS, loss)
        return {
            "loss": loss
        }

    def _calc_loss(self,
                   logits: torch.Tensor,
                   target_tensor: torch.Tensor
                   ) -> torch.Tensor:
        return self.loss_function(target_tensor, logits)


    def predict_step(self,
                     batch: Dict[str, torch.Tensor],
                     batch_idx: int,
                     dataloader_idx: Optional[int] = None
                     ) -> torch.Tensor:
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]     # BS x S
        target = batch[TARGET_ENTRY_NAME]  # BS

        logits = self(batch, batch_idx)  # BS x S x I

        target_logits = self._extract_target_logits(input_seq, logits)
        return target_logits

    def validation_step(self,
                        batch: Dict[str, torch.Tensor],
                        batch_idx: int
                        ) -> Dict[str, torch.Tensor]:
        """
        Performs a validation step on a batch of sequences and returns the entries according
        to `build_eval_step_return_dict`.

        `batch` must be a dictionary containing the following entries:
            * `ITEM_SEQ_ENTRY_NAME`: a tensor of size (N, S),
            * `TARGET_ENTRY_NAME`: a tensor of size (N) with the target items,

        :param batch: the batch
        :param batch_idx: the batch number.
        :return: A dictionary with entries according to `build_eval_step_return_dict`.

        where N is the batch size and S the max sequence length.
        """

        input_seq = batch[ITEM_SEQ_ENTRY_NAME]     # BS x S
        target = batch[TARGET_ENTRY_NAME]  # BS

        logits = self(batch, batch_idx)  # BS x S x I

        target_logits = self._extract_target_logits(input_seq, logits)

        loss = self._calc_loss(target_logits, target)
        self.log(LOG_KEY_VALIDATION_LOSS, loss, prog_bar=True)

        mask = None if len(target.size()) == 1 else ~ target.eq(self.item_tokenizer.pad_token_id)
        return build_eval_step_return_dict(input_seq, target_logits, target, mask=mask)

    def _extract_target_logits(self, input_seq: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Finds the model output for the last input item in each sequence.

        :param input_seq: the input sequence. [BS x S]
        :param logits: the logits [BS x S x I]

        :return: the logits for the last input item of the sequence. [BS x I]
        """
        # calculate the padding mask where each non-padding token has the value `1`
        padding_mask = get_padding_mask(input_seq, self.item_tokenizer)  # [BS x S]
        seq_length = padding_mask.sum(dim=-1) - 1  # [BS]

        batch_index = torch.arange(input_seq.size()[0])  # [BS]

        # select only the outputs at the last step of each sequence
        target_logits = logits[batch_index, seq_length]  # [BS, I]

        return target_logits

    def predict_step(self,
                     batch: Dict[str, torch.Tensor],
                     batch_idx: int,
                     dataloader_idx: Optional[int] = None
                     ) -> torch.Tensor:

        input_seq = batch[ITEM_SEQ_ENTRY_NAME]     # BS x S
        logits = self(batch, batch_idx)  # BS x S x I
        target_logits = self._extract_target_logits(input_seq, logits)
        return target_logits


    def _extract_target_item_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.user_key_len > 0:
            return logits[:,1:,:]
        else:
            return logits

