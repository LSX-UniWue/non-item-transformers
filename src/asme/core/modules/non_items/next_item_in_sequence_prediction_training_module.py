from typing import Union, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from asme.core.init.factories.features.tokenizer_factory import TOKENIZERS_PREFIX
from asme.core.losses.losses import DEFAULT_REDUCTION
from asme.core.metrics.container.metrics_container import MetricsContainer
from asme.core.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.core.modules import LOG_KEY_VALIDATION_LOSS, LOG_KEY_TRAINING_LOSS
from asme.core.modules.next_item_prediction_training_module import BaseNextItemPredictionTrainingModule
from asme.core.modules.util.module_util import get_padding_mask, build_eval_step_return_dict, build_model_input, \
    convert_target_to_multi_hot
from asme.core.tokenization.tokenizer import Tokenizer
from asme.core.utils.hyperparameter_utils import save_hyperparameters
from asme.core.utils.inject import InjectTokenizer, inject, InjectTokenizers
from asme.data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME, TARGET_SUFFIX, KEY_DELIMITER


class NextItemInSequencePredictionTrainingModule(BaseNextItemPredictionTrainingModule):

    @inject(
        item_tokenizer=InjectTokenizer(ITEM_SEQ_ENTRY_NAME),
        tokenizers=InjectTokenizers()
    )
    @save_hyperparameters
    def __init__(self,
                 model: SequenceRecommenderModel,
                 item_tokenizer: Tokenizer,
                 tokenizers: Dict[str, Tokenizer],
                 metrics: MetricsContainer,
                 learning_rate: float = 0.001,
                 beta_1: float = 0.99,
                 beta_2: float = 0.998,
                 weight_decay: float = 0,
                 loss_category: str = None,
                 loss_factor: float = 1,
                 loss_category_epochs: int = None,
                 item_type_id: str = None,
                 use_item_cat_loss: bool = True,
                 validation_metrics_on_item: bool = True,
                 first_item: bool = False
                 ):

        cat_tokenizer = tokenizers.get(TOKENIZERS_PREFIX + KEY_DELIMITER + loss_category)
        loss = ItemCategoryMixedLoss(item_tokenizer, cat_tokenizer=cat_tokenizer)
        super().__init__(model=model, metrics=metrics, learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2,
                         weight_decay=weight_decay, loss_function=loss)
        self.user_key_len = len(model.optional_metadata_keys())
        self.first_item = first_item
        self.loss_category = loss_category
        self.item_cat_loss = use_item_cat_loss
        self.loss = loss
        self.cat_tokenizer = cat_tokenizer
        self.item_tokenizer = item_tokenizer
        self.loss_factor = loss_factor
        self.validation_on_item = validation_metrics_on_item
        self.loss_category_epochs = loss_category_epochs
        self.item_type_id = item_type_id
        self.save_hyperparameters(self.hyperparameters)


    def forward(self,
                batch: Dict[str, torch.Tensor],
                batch_idx: Optional[int] = None
                ) -> torch.Tensor:
        input_data = build_model_input(self.model, self.item_tokenizer, batch)
        # call the model
        return self.model(input_data)


    def on_train_epoch_start(self):
        if self.loss_category_epochs is not None and self.current_epoch >= self.loss_category_epochs:
            self.loss_factor = 0

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

        item_logits, cat_logits = self(batch, batch_idx)
        item_target = batch[TARGET_ENTRY_NAME]
        cat_target = batch[self.loss_category + TARGET_SUFFIX]

        if not self.first_item:
            item_logits = self._extract_target_item_logits(item_logits)
            cat_logits = self._extract_target_item_logits(cat_logits)

        #Set item target to padding id for nonitems to exclude them from the gradient
        if self.item_type_id is not None:
            item_type_id_target = batch[self.item_type_id]
            item_target[item_type_id_target == 1] = self.item_tokenizer.pad_token_id
        if self.item_cat_loss is False:
            #set cat target to padding id for items to exclude this from the gradient
            pad_tensor = torch.tensor([self.cat_tokenizer.pad_token_id]*cat_target.shape[2], device=cat_target.device).type(torch.LongTensor)
            cat_target[item_type_id_target == 0] = pad_tensor

        item_loss = self.loss.item_forward(item_target, item_logits)
        cat_loss = self.loss.cat_forward(cat_target, cat_logits)

        overall_loss = self.combine_losses(cat_loss, item_loss)

        self.log(LOG_KEY_TRAINING_LOSS, overall_loss)
        self.log("unsmoothed_" + LOG_KEY_TRAINING_LOSS, overall_loss, prog_bar=True)
        self.log("item_" + LOG_KEY_TRAINING_LOSS, item_loss, prog_bar=True)
        self.log("cat_" + LOG_KEY_TRAINING_LOSS, cat_loss, prog_bar=True)
        return {
            "loss": overall_loss
        }

    def combine_losses(self, cat_loss, item_loss):
        return item_loss + self.loss_factor * cat_loss

    def predict_step(self,
                     batch: Dict[str, torch.Tensor],
                     batch_idx: int,
                     dataloader_idx: Optional[int] = None
                     ) -> torch.Tensor:
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]  # BS x S

        item_logits, cat_logits = self(batch, batch_idx)  # BS x S x I

        target_logits = self._extract_target_logits(input_seq, item_logits)
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

        item_input_seq = batch[ITEM_SEQ_ENTRY_NAME]  # BS x S
        cat_input_seq = batch[self.loss_category]

        item_target = batch[TARGET_ENTRY_NAME]  # BS
        cat_target = batch[self.loss_category + TARGET_SUFFIX]  # BS

        item_logits, cat_logits = self(batch, batch_idx)  # BS x S x I

        item_target_logits = self._extract_target_logits(item_input_seq, item_logits)
        cat_target_logits = self._extract_target_logits(item_input_seq, cat_logits)

        item_loss = self.loss.item_forward(item_target, item_target_logits)
        cat_loss = self.loss.cat_forward(cat_target, cat_target_logits)

        overall_loss = self.combine_losses(cat_loss, item_loss)
        self.log(LOG_KEY_VALIDATION_LOSS, overall_loss, prog_bar=True)
        self.log("unsmoothed_" + LOG_KEY_VALIDATION_LOSS, overall_loss, prog_bar=True)
        self.log("item_" + LOG_KEY_VALIDATION_LOSS, item_loss,prog_bar=True)
        self.log("cat_" + LOG_KEY_VALIDATION_LOSS, cat_loss,prog_bar=True)

        if self.validation_on_item:
            mask = None if len(item_target.size()) == 1 else ~ item_target.eq(self.item_tokenizer.pad_token_id)
            return build_eval_step_return_dict(item_input_seq, item_target_logits, item_target, mask=mask)
        else:
            mask = None if len(cat_target.size()) == 1 else ~ cat_target.eq(self.cat_tokenizer.pad_token_id)
            return build_eval_step_return_dict(cat_input_seq, cat_target_logits, cat_target, mask=mask)


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

    def _extract_target_item_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.user_key_len > 0:
            return logits[:, 1:, :]
        else:
            return logits


class ItemCategoryMixedLoss(nn.Module):

    @save_hyperparameters
    def __init__(self, item_tokenizer, cat_tokenizer, reduction: str = DEFAULT_REDUCTION):
        super().__init__()
        self.reduction = reduction
        self.item_tokenizer = item_tokenizer
        self.cat_tokenizer = cat_tokenizer

    def item_forward(self, target: torch.Tensor, logit: torch.Tensor) -> torch.Tensor:

        # SASRecFullSequenceCrossEntropyLoss
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.item_tokenizer.pad_token_id)
        logit_size = logit.size()
        target_size = target.size()

        # we need to accomodate both training with the full target sequence and validation with single target
        if isinstance(target_size, torch.Size) and len(target_size) > 1:
            # we have a full target sequence
            # logit has size: N, S, I needs to be: N*S,I
            logit = torch.reshape(logit, [-1, logit_size[2]])

            # target has size: N, S needs to be: N*S
            target = torch.reshape(target, [-1])

            return loss_fn(logit, target)
        else:
            return loss_fn(logit, target)

    def cat_forward(self, target: torch.Tensor, logit: torch.Tensor) -> torch.Tensor:
        logit_size = logit.size()
        target_size = target.size()

        # we need to accomodate both training with the full target sequence and validation with single targets
        if isinstance(target_size, torch.Size) and len(target_size) > 2:
            # we have a full target sequence
            # logit has size: N, S, I needs to be: N*S,I
            logit = torch.reshape(logit, [-1, logit_size[2]])

            # target has size: N, S needs to be: N*S
            target = convert_target_to_multi_hot(target, len(self.cat_tokenizer), self.cat_tokenizer.pad_token_id)

            target = torch.reshape(target, [-1, logit_size[2]])
            return F.binary_cross_entropy_with_logits(logit, target)

        else:
            target = convert_target_to_multi_hot(target, len(self.cat_tokenizer), self.cat_tokenizer.pad_token_id)

            return F.binary_cross_entropy_with_logits(logit, target)
