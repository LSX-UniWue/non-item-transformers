from typing import Dict, Any, List

from asme.core.init.factories.features.item_dictionary_factory import get_dict_key_for_attribute
from asme.core.tokenization.item_dictionary import SpecialValues
from asme.data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME, TARGET_SUFFIX
from asme.data.datasets.processors.processor import Processor
from asme.core.init.factories.features.tokenizer_factory import get_tokenizer_key_for_voc, \
    ITEM_TOKENIZER_ID
from asme.core.tokenization.tokenizer import Tokenizer
from asme.data.datasets.processors.utils import random_uniform, random_, get_tokenizer, get_mask_value
import random


class ClozeMaskProcessor(Processor):
    """
    A processor, that replaces with a given probability items in the sequence
    with a mask token that the model should than predict (e.g. BERT4Rec)

    Example:
        Input:
            session: [1, 5, 7, 8]
        Output:
            session:          [1, 5, 101, 8]
            targets:          [0, 0, 7,   0]

        where 101 is the mask token id
        please use 0 in the target for loss masking

    """

    def __init__(self,
                 tokenizers: Dict[str, Tokenizer],
                 special_values: Dict[str, SpecialValues],
                 mask_prob: float,
                 only_last_item_mask_prob: float,
                 masking_targets: List[str] = None
                 ):
        """
        :param tokenizers: the tokenizers
        :param mask_prob: the mask prob to use for masking items in the sequence
        :param only_last_item_mask_prob: the prob that the last item in the sequence should only be masked
        """
        super().__init__()

        if masking_targets is None:
            masking_targets = [ITEM_SEQ_ENTRY_NAME]
        if ITEM_SEQ_ENTRY_NAME not in masking_targets:
            masking_targets.append(ITEM_SEQ_ENTRY_NAME)

        self.tokenizers = tokenizers
        self.special_values = special_values

        self.mask_prob = mask_prob
        self.only_last_item_mask_prob = only_last_item_mask_prob
        self.masking_targets = masking_targets

    def process(self,
                parsed_sequence: Dict[str, Any]
                ) -> Dict[str, Any]:
        sequence = parsed_sequence[ITEM_SEQ_ENTRY_NAME]


        sequences = {
            mask_target: parsed_sequence[mask_target] for mask_target in self.masking_targets
        }
        targets = {
            mask_target: parsed_sequence[mask_target].copy() for mask_target in self.masking_targets
        }

        # first we decide if we only mask the last item
        mask_last_item_prob = random_uniform()
        if mask_last_item_prob <= self.only_last_item_mask_prob:
            for mask_target, sequence in sequences.items():
                target = targets[mask_target]
                tokenizer = get_tokenizer(self.tokenizers, mask_target)

                last_item = len(sequence) - 1
                sequence[last_item] = get_mask_value(self.tokenizers, self.special_values, mask_target, sequence)

                # if it is the original sequence, update the target and set it to the pad token id
                if mask_target == ITEM_SEQ_ENTRY_NAME:
                    padding_mask = tokenizer.pad_token_id
                else:
                    padding_mask = self.tokenizers[mask_target].pad_token_id
                target[:last_item] = [padding_mask] * last_item
        else:
            for index in range(0, len(sequence)):
                prob = random_uniform()
                for mask_target, sequence in sequences.items():
                    target = targets[mask_target]
                    if prob < self.mask_prob:
                        prob = prob / self.mask_prob

                        if prob < 0.8:
                            sequence[index] = get_mask_value(self.tokenizers, self.special_values, mask_target, sequence)
                        elif prob < 0.9:
                            tokenizer = get_tokenizer(self.tokenizers, mask_target)
                            if tokenizer:
                                random_index = random_(0, len(get_tokenizer(self.tokenizers, mask_target)) - 1)
                                sequence[index] = [random_index] if isinstance(sequence[0], list) else random_index
                    else:
                        # we use the padding token for masking the cross entropy loss
                        loss_mask = self.tokenizers[get_tokenizer_key_for_voc(ITEM_TOKENIZER_ID)].pad_token_id
                        target[index] = [loss_mask] if isinstance(sequence[0], list) else loss_mask


        for mask_target, sequence in sequences.items():
            parsed_sequence[mask_target+TARGET_SUFFIX] = targets[mask_target]

        return parsed_sequence
