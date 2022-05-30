from typing import Union, Dict, Optional

import torch

import pytorch_lightning as pl
from asme.core.losses.losses import SequenceRecommenderContrastiveLoss
from asme.core.models.common.layers.data.sequence import InputSequence

from asme.core.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.core.modules import LOG_KEY_TRAINING_LOSS
from asme.core.utils.inject import InjectTokenizer, inject
from asme.data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME, POSITIVE_SAMPLES_ENTRY_NAME, \
    NEGATIVE_SAMPLES_ENTRY_NAME
from asme.core.losses.sasrec.sas_rec_losses import SASRecBinaryCrossEntropyLoss
from asme.core.metrics.container.metrics_container import MetricsContainer
from asme.core.modules.metrics_trait import MetricsTrait
from asme.core.modules.util.module_util import get_padding_mask, build_eval_step_return_dict, get_additional_meta_data
from asme.core.tokenization.tokenizer import Tokenizer
from asme.core.utils.hyperparameter_utils import save_hyperparameters


class SequenceNextItemPredictionTrainingModule(MetricsTrait, pl.LightningModule):

    """
    the module for training a model using a sequence and positive and negative items based on the sequence

    models that can be trained with this module are:
    - SASRec
    - Caser
    - CosRec
    - HGN
    """

    @inject(item_tokenizer=InjectTokenizer("item"))
    @save_hyperparameters
    def __init__(self,
                 model: SequenceRecommenderModel,
                 item_tokenizer: Tokenizer,
                 metrics: MetricsContainer,
                 learning_rate: float = 0.001,
                 beta_1: float = 0.99,
                 beta_2: float = 0.998,
                 weight_decay: float = 1e-3,
                 loss_function: SequenceRecommenderContrastiveLoss = SASRecBinaryCrossEntropyLoss()
                 ):
        """
        inits the training module
        :param model: the model to train
        :param learning_rate: the learning rate
        :param beta_1: the beta1 of the adam optimizer
        :param beta_2: the beta2 of the adam optimizer
        :param item_tokenizer: the tokenizer
        :param metrics: metrics to compute on validation/test
        """
        super().__init__()
        self.model = model

        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay

        self.item_tokenizer = item_tokenizer
        self.metrics = metrics

        self.loss_function = loss_function

        self.save_hyperparameters(self.hyperparameters)

    def get_metrics(self) -> MetricsContainer:
        return self.metrics

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int
                      ) -> Optional[Union[torch.Tensor, Dict[str, Union[torch.Tensor, float]]]]:
        """
        Performs a training step on a batch of sequences and returns the overall loss.

        `batch` must be a dictionary containing the following entries:
            * `ITEM_SEQ_ENTRY_NAME`: a tensor of size (N, S),
            * `POSITIVE_SAMPLES_ENTRY_NAME`: a tensor of size (N) containing the next sequence items (pos examples),
            * `NEGATIVE_SAMPLES_ENTRY_NAME`: a tensor of size (N) containing a negative item (sampled)

        Where N is the batch size and S the max sequence length.

        A padding mask will be generated on the fly, and also the masking of items

        :param batch: the batch
        :param batch_idx: the batch number.
        :return: the total loss
        """

        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        padding_mask = get_padding_mask(input_seq, self.item_tokenizer)

        pos = batch[POSITIVE_SAMPLES_ENTRY_NAME]
        neg = batch[NEGATIVE_SAMPLES_ENTRY_NAME]

        # add users and other meta data
        additional_meta_data = get_additional_meta_data(self.model, batch)

        additional_meta_data["positive_samples"] = pos
        additional_meta_data["negative_samples"] = neg

        input_sequence = InputSequence(input_seq, padding_mask, additional_meta_data)
        pos_logits, neg_logits = self.model(input_sequence)

        # calculate the item mask (same as the padding mask in the non basket-recommendation setting)
        item_mask = input_seq.ne(self.item_tokenizer.pad_token_id)
        loss = self._calc_loss(pos_logits, neg_logits, item_mask)
        self.log(LOG_KEY_TRAINING_LOSS, loss)
        return {
            "loss": loss
        }

    def _calc_loss(self,
                   pos_logits: torch.Tensor,
                   neg_logits: torch.Tensor,
                   padding_mask: torch.Tensor
                   ) -> torch.Tensor:
        return self.loss_function(pos_logits, neg_logits, mask=padding_mask)

    def validation_step(self,
                        batch: Dict[str, torch.Tensor],
                        batch_idx: int
                        ) -> Dict[str, torch.Tensor]:
        """
        Performs a validation step on a batch of sequences and returns the overall loss.

        `batch` must be a dictionary containing the following entries:
            * `ITEM_SEQ_ENTRY_NAME`: a tensor of size (N, S),
            * `TARGET_ENTRY_NAME`: a tensor of size (N) containing the next item of the provided sequence

        Where N is the batch size and S the max sequence length.

        A padding mask will be generated on the fly, and also the masking of items

        :param batch: the batch
        :param batch_idx: the batch number.
        :return: A dictionary with entries according to `build_eval_step_return_dict`.
        """
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        targets = batch[TARGET_ENTRY_NAME]
        prediction = self.predict(batch, batch_idx)

        mask = None if len(targets.size()) == 1 else ~ targets.eq(self.item_tokenizer.pad_token_id)
        return build_eval_step_return_dict(input_seq, prediction, targets, mask=mask)

    def test_step(self,
                  batch: Dict[str, torch.Tensor],
                  batch_idx: int
                  ) -> Dict[str, torch.Tensor]:
        return self.validation_step(batch, batch_idx)

    def predict_step(self,
                batch: Dict[str, torch.Tensor],
                batch_idx: int,
                dataloader_idx: Optional[int] = None
                ) -> torch.Tensor:
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]

        # add users and other meta data
        additional_meta_data = get_additional_meta_data(self.model, batch)

        # calc the padding mask
        padding_mask = get_padding_mask(input_seq, self.item_tokenizer)

        # provide items that the target item will be ranked against
        # TODO (AD) refactor this into a composable class to allow different strategies for item selection
        batch_size = input_seq.size()[0]
        device = input_seq.device
        items_to_rank = torch.as_tensor(self.item_tokenizer.get_vocabulary().ids(), dtype=torch.long, device=device)
        items_to_rank = items_to_rank.repeat([batch_size, 1])

        additional_meta_data["positive_samples"] = items_to_rank

        input_sequence = InputSequence(input_seq, padding_mask, additional_meta_data)
        return self.model(input_sequence)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.learning_rate,
                                betas=(self.beta_1, self.beta_2),
                                weight_decay=self.weight_decay)
