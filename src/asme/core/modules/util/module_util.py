from typing import Dict

import torch
from torch.nn import functional as F

from asme.core.models.common.layers.data.sequence import InputSequence
from asme.core.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.core.modules.constants import RETURN_KEY_PREDICTIONS, RETURN_KEY_TARGETS, RETURN_KEY_MASK, RETURN_KEY_SEQUENCE
from asme.core.tokenization.tokenizer import Tokenizer
from asme.data.datasets import ITEM_SEQ_ENTRY_NAME


def get_padding_mask(sequence: torch.Tensor,
                     tokenizer: Tokenizer
                     ) -> torch.Tensor:
    """
    generates the padding mask based on the tokenizer for the provided sequence

    :param sequence: the sequence tensor :math`(N, S)`
    :param tokenizer: the tokenizer
    :return: the padding mask True where the sequence was not padded, False where the sequence was padded :math`(N, S)`

    where N is the batch size and S the max sequence length
    """

    if len(sequence.size()) > 2:
        sequence = sequence.max(dim=2).values

    # the masking should be true where the padding token is set
    return sequence.ne(tokenizer.pad_token_id)


def convert_target_for_multi_label_margin_loss(target: torch.Tensor,
                                               num_classes: int,
                                               pad_token_id: int
                                               ) -> torch.Tensor:
    """
    pads the (already padded) target vector the the number of classes,
    sets every pad token to a negative value (e.g. required for the MultiLabelMarginLoss of PyTorch

    :param target: the target tensor, containing target class ids :math `(N, X)`
    :param num_classes: the number of classes for the multi-class multi-classification
    :param pad_token_id: the padding token id
    :return: a tensor of shape :math `(N, C)`

    where C is the number of classes, N the batch size, and X the padded class id length
    """
    converted_target = F.pad(target, [0, num_classes - target.size()[1]], value=pad_token_id)
    # TODO: also mask all special tokens
    converted_target[converted_target == pad_token_id] = -1
    return converted_target


def convert_target_to_multi_hot(target_tensor: torch.Tensor,
                                num_classes: int,
                                pad_token_id: int
                                ) -> torch.Tensor:
    """
    generates a mulit-hot vector of the provided indices in target_tensor

    :param target_tensor: a target tensor with multiple target for every sequence. [BS x S]
    :param num_classes: the number of target classes.
    :param pad_token_id: the id of the padding token

    :return: a tensor [BS, I] with a multi-hot coded target vector for each sequence. Target items are signaled by `1`.
    """

    multi_hot = torch.zeros(list(target_tensor.size()[:-1]) + [num_classes], device=target_tensor.device)
    target = target_tensor.clone()
    target[target == -100] = pad_token_id  # FIXME (AD): It seems that a target value of -100 has a special meaning, but I don't know where this comes from ?

    multi_hot.scatter_(-1, target, 1)
    # remove the padding for each multi-hot target
    multi_hot[..., pad_token_id] = 0
    return multi_hot


def build_eval_step_return_dict(input_sequence: torch.Tensor,
                                predictions: torch.Tensor,
                                targets: torch.Tensor,
                                mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:

    """
    Generates a dictionary to be returned from the validation/test step which contains information to be provided to callbacks.

    :param input_sequence: the input sequence used by the model to calculate the predictions.
    :param predictions: Predictions made by the model in the current step.
    :param targets: Expected outputs from the model in the current step.
    :param mask: Optional mask which is forwarded to metrics.

    :returns: A dictionary containing the values provided to this function using the keys defined in modules/util/constants.py.
    """

    return_dict = {
        RETURN_KEY_SEQUENCE: input_sequence,
        RETURN_KEY_PREDICTIONS: predictions,
        RETURN_KEY_TARGETS: targets,
    }

    if mask is not None:
        return_dict.update({RETURN_KEY_MASK: mask})

    return return_dict


def get_additional_meta_data(model: SequenceRecommenderModel, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Collects all relevant additional metadata from the batch, according to `SequenceRecommendationModel.additional_metadata_keys(self)`) and
    returns it as a dictionary.

    :param model: the recommender model
    :param batch: the full batch
    :return: a dictionary with the collected additional metadata used by this model.
    :raises Exception: if an entry for a relevant additional metadata item can not be found in the batch.
    """
    required_metadata_keys = model.required_metadata_keys()

    metadata = {}
    for key in required_metadata_keys:
        if key not in batch:
            raise Exception(f"The batch does not contain the following additional metadata: {key}. "
                            f"Found the following batch entries: {', '.join(batch.keys())}")
        metadata[key] = batch[key]

    optional_metadata_keys = model.optional_metadata_keys()
    for key in optional_metadata_keys:
        if key in batch:
            metadata[key] = batch[key]

    return metadata


def build_model_input(model: SequenceRecommenderModel,
                      item_tokenizer: Tokenizer,
                      batch: Dict[str, torch.Tensor]
                      ) -> InputSequence:
    """
    builds a simple input sequence for the model

    :param model:
    :param item_tokenizer:
    :param batch:
    :return: A input sequence object with all the data provided by the batch
    """
    input_seq = batch[ITEM_SEQ_ENTRY_NAME]

    # calc the padding mask
    padding_mask = get_padding_mask(sequence=input_seq, tokenizer=item_tokenizer)
    additional_metadata = get_additional_meta_data(model, batch)

    return InputSequence(input_seq, padding_mask, additional_metadata)
