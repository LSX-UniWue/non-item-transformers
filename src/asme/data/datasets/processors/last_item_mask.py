from typing import Dict, Any, List

from asme.core.tokenization.item_dictionary import SpecialValues
from asme.core.tokenization.special_values import SpecialValues
from asme.data.datasets import ITEM_SEQ_ENTRY_NAME
from asme.data.datasets.processors.processor import Processor
from asme.core.tokenization.tokenizer import Tokenizer
from asme.data.datasets.processors.utils import get_mask_value


class LastItemMaskProcessor(Processor):
    """
    Adds a mask token at the end of the input sequence.
    This is useful for evaluation purposes in some models, e.g. BERT4Rec.

    Example:
        Input:
            session: [1, 5, 7, 8]
        Output:
            session:          [1, 5, 7, 8, 101]

    where 101 is the mask token id
    """

    def __init__(self,
                 tokenizers: Dict[str, Tokenizer],
                 special_values: Dict[str, SpecialValues],
                 masking_targets: List[str] = None
                 ):
        super().__init__()

        if masking_targets is None:
            masking_targets = [ITEM_SEQ_ENTRY_NAME]

        self.tokenizers = tokenizers
        self.special_values = special_values
        self.masking_targets = masking_targets

    def process(self,
                parsed_sequence: Dict[str, Any]
                ) -> Dict[str, Any]:

        for target in self.masking_targets:
            sequence = parsed_sequence[target]
            mask_token = get_mask_value(self.tokenizers, self.special_values, target, sequence)
            sequence.append(mask_token)

        return parsed_sequence
