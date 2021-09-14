from abc import abstractmethod
from typing import Union, Dict, Optional

import torch

import pytorch_lightning as pl
from asme.core.losses.losses import CrossEntropyLoss

from asme.core.models.common.layers.data.sequence import InputSequence
from asme.core.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME, POSITIVE_SAMPLES_ENTRY_NAME, \
    NEGATIVE_SAMPLES_ENTRY_NAME
from asme.core.metrics.container.metrics_container import MetricsContainer
from asme.core.modules import LOG_KEY_VALIDATION_LOSS, LOG_KEY_TRAINING_LOSS
from asme.core.modules.metrics_trait import MetricsTrait
from asme.core.modules.util.module_util import get_padding_mask, build_eval_step_return_dict, get_additional_meta_data
from asme.core.tokenization.tokenizer import Tokenizer
from asme.core.utils.hyperparameter_utils import save_hyperparameters


class BaseNextItemPredictionTrainingModule(MetricsTrait, pl.LightningModule):
    """
    A training module for models that get a sequence and must predict the next item in the sequence

    Supported models are:
    - NARM
    - RNN
    """

    @save_hyperparameters
    def __init__(self,
                 model: SequenceRecommenderModel,
                 item_tokenizer: Tokenizer,
                 metrics: MetricsContainer,
                 learning_rate: float = 0.001,
                 beta_1: float = 0.99,
                 beta_2: float = 0.998,
                 weight_decay: float = 0,
                 loss_function: Optional = None
                 ):
        """
        Initializes the training module.
        """
        super().__init__()
        self.model = model

        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay

        self.item_tokenizer = item_tokenizer

        self.metrics = metrics

        if loss_function is None:
            loss_function = CrossEntropyLoss
        self.loss_function = loss_function(item_tokenizer)

        self.save_hyperparameters(self.hyperparameters)

    def get_metrics(self) -> MetricsContainer:
        return self.metrics

    def forward(self,
                batch: Dict[str, torch.Tensor],
                batch_idx: int
                ) -> torch.Tensor:
        """
        Applies the recommender model on a batch of sequences and returns logits for every sample in the batch.

        `batch` must be a dictionary containing the following entries:
            * `ITEM_SEQ_ENTRY_NAME`: a tensor of size (N, S),

        A padding mask will be calculated on the fly, based on the `self.tokenizer` of the module.

        :param batch: a batch.
        :param batch_idx: the batch number.

        :return: a tensor with logits for every batch of size (N, I)

        Where N is the batch size, S the max sequence length, and I the item vocabulary size.
        """
        additional_meta_data = get_additional_meta_data(self.model, batch)

        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        padding_mask = get_padding_mask(input_seq, self.item_tokenizer)

        sequence = InputSequence(input_seq, padding_mask, additional_meta_data)
        return self.model(sequence)

    @abstractmethod
    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int
                      ) -> Optional[Union[torch.Tensor, Dict[str, Union[torch.Tensor, float]]]]:
        pass

    @abstractmethod
    def validation_step(self,
                        batch: Dict[str, torch.Tensor],
                        batch_idx: int
                        ) -> Dict[str, torch.Tensor]:
        pass

    def test_step(self,
                  batch: Dict[str, torch.Tensor],
                  batch_idx: int
                  ) -> Dict[str, torch.Tensor]:
        return self.validation_step(batch, batch_idx)

    def predict(self,
                batch: Dict[str, torch.Tensor],
                batch_idx: int,
                dataloader_idx: Optional[int] = None
                ) -> torch.Tensor:
        return self(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.learning_rate,
                                betas=(self.beta_1, self.beta_2),
                                weight_decay=self.weight_decay)


class NextItemPredictionTrainingModule(BaseNextItemPredictionTrainingModule):

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

        loss = self._calc_loss(logits, target)
        self.log(LOG_KEY_TRAINING_LOSS, loss)
        return {
            "loss": loss
        }

    def _calc_loss(self,
                   logits: torch.Tensor,
                   target_tensor: torch.Tensor
                   ) -> torch.Tensor:
        return self.loss_function(target_tensor, logits)

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

        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        target = batch[TARGET_ENTRY_NAME]

        logits = self(batch, batch_idx)

        loss = self._calc_loss(logits, target)
        self.log(LOG_KEY_VALIDATION_LOSS, loss, prog_bar=True)

        mask = None if len(target.size()) == 1 else ~ target.eq(self.item_tokenizer.pad_token_id)
        return build_eval_step_return_dict(input_seq, logits, target, mask=mask)


class NextItemPredictionWithNegativeSampleTrainingModule(BaseNextItemPredictionTrainingModule):

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int
                      ) -> Optional[Union[torch.Tensor, Dict[str, Union[torch.Tensor, float]]]]:
        """
        Performs a training step on a batch of sequences and returns the overall loss.

        `batch` must be a dictionary containing the following entries:
            * `ITEM_SEQ_ENTRY_NAME`: a tensor of size (N, S, M),
            * `POSITIVE_SAMPLES_ENTRY_NAME`: a tensor of size (N, M) containing the next sequence items (pos examples)
            * `NEGATIVE_SAMPLES_ENTRY_NAME`: a tensor of size (N, M) containing a negative item (sampled)

        Where N is the batch size, S the max sequence length and M the max items per sequence step.

        :param batch: the batch
        :param batch_idx: the batch number.
        :return: the total loss
        """
        pos_items = batch[POSITIVE_SAMPLES_ENTRY_NAME]
        neg_items = batch[NEGATIVE_SAMPLES_ENTRY_NAME]

        logits = self.predict(batch, batch_idx)

        loss = self._calc_loss(logits, pos_items, neg_items)
        return {
            'loss': loss
        }

    def _calc_loss(self,
                   logit: torch.Tensor,
                   pos_items: torch.Tensor,
                   neg_items: torch.Tensor
                   ) -> torch.Tensor:
        return self.loss_function(logit, pos_items, neg_items)

    def validation_step(self,
                        batch: Dict[str, torch.Tensor],
                        batch_idx: int
                        ) -> Dict[str, torch.Tensor]:
        """
        Performs a validation step on a batch of sequences and returns the overall loss.

        `batch` must be a dictionary containing the following entries:
            * `ITEM_SEQ_ENTRY_NAME`: a tensor of size (N, S, M),
            * `TARGET_ENTRY_NAME`: a tensor of size (N, M) with the target items,

        A padding mask will be generated on the fly, and also the masking of items

        Where N is the batch size, S the max sequence length and M the max items per sequence step.

        :param batch: the batch
        :param batch_idx: the batch number.
        :return: A dictionary with entries according to `build_eval_step_return_dict`.
        """
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        targets = batch[TARGET_ENTRY_NAME]

        prediction = self.predict(batch, batch_idx)

        mask = None if len(targets.size()) == 1 else ~ targets.eq(self.item_tokenizer.pad_token_id)
        return build_eval_step_return_dict(input_seq, prediction, targets, mask=mask)
