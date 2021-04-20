from typing import Dict

import torch
from torch.nn import functional as F

from asme.modules.constants import RETURN_KEY_PREDICTIONS, RETURN_KEY_TARGETS, RETURN_KEY_MASK, RETURN_KEY_SEQUENCE
from asme.tokenization.tokenizer import Tokenizer


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

    :param target_tensor:
    :param num_classes:
    :param pad_token_id:
    :return: a tensor with 1s in the indices specified by
    """

    multi_hot = torch.zeros(list(target_tensor.size()[:-1]) + [num_classes], device=target_tensor.device)
    target = target_tensor.clone()
    target[target == -100] = pad_token_id

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