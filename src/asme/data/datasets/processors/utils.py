from typing import Dict, Union, List

import torch

from asme.core.init.factories.features.special_values_feature_factory import get_dict_key_for_attribute
from asme.core.init.factories.features.tokenizer_factory import ITEM_TOKENIZER_ID, get_tokenizer_key_for_voc
from asme.core.tokenization.tokenizer import Tokenizer
from asme.core.tokenization.item_dictionary import SpecialValues
from asme.data.datasets import ITEM_SEQ_ENTRY_NAME


def get_mask_value(tokenizers: Dict[str, Tokenizer],
                   special_values: Dict[str, SpecialValues],
                   target: str,
                   sequence: Union[List[int], List[List[int]]]
                   ) -> Union[int, List[int]]:
    tokenizer = get_tokenizer(tokenizers, target)
    if tokenizer is not None:
        mask_value = tokenizer.mask_token_id
        return [mask_value] if isinstance(sequence[0], list) else mask_value
    special_values = special_values.get(get_dict_key_for_attribute(target), None)
    if special_values is not None:
        return special_values.get_mask_value()

def get_tokenizer(tokenizers: Dict[str, Tokenizer],
                  target: str
                  ) -> Tokenizer:
    tokenizer_id = target
    if target == ITEM_SEQ_ENTRY_NAME:
        tokenizer_id = ITEM_TOKENIZER_ID

    return tokenizers.get(get_tokenizer_key_for_voc(tokenizer_id), None)



def random_uniform(start: float = 0., end: float = 1.) -> float:
    """
    Draws a single random number uniformly from a continuous distribution (pytorch) in [start; end).

    :param start: lowest number
    :param end: highest number

    :return: a single float from [start; end).
    """
    return torch.empty((), dtype=torch.float, device="cpu").uniform_(start, end).item()


def random_(start: int, end: int) -> int:
    """
    Draws uniformly from a discrete distribution in [start; end]

    :param start: lowest number.
    :param end: highest number.

    :return: a single number.
    """
    return torch.empty((), dtype=torch.int, device="cpu").random_(start, end).item()
