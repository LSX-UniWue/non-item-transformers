from typing import Dict, Any, List

from asme.core.tokenization.vector_dictionary import ItemDictionary
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
                 dictionaries: Dict[str, ItemDictionary],
                 masking_targets: List[str] = None
                 ):
        super().__init__()

        if masking_targets is None:
            masking_targets = [ITEM_SEQ_ENTRY_NAME]

        self.tokenizers = tokenizers
        self.vector_dictionary = dictionaries
        self.masking_targets = masking_targets

    def process(self,
                parsed_sequence: Dict[str, Any]
                ) -> Dict[str, Any]:

        for target in self.masking_targets:
            sequence = parsed_sequence[target]
            mask_token = get_mask_value(self.tokenizers, self.vector_dictionary, target, sequence)
            sequence.append(mask_token)

        return parsed_sequence
