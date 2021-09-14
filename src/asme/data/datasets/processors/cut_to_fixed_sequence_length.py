from typing import Dict, Any

from asme.data.datasets import ITEM_SEQ_ENTRY_NAME
from asme.data.datasets.processors.processor import Processor


class CutToFixedSequenceLengthProcessor(Processor):
    """
    Cuts the sequence to a fixed length

    Example:
        Input:
            session: [1, 5, 7, 8]
        Output:
            session:          [1, 5, 7, 8, 101]

    where 101 is the mask token id
    """

    def __init__(self,
                 fixed_length: int
                 ):
        super().__init__()
        self.fixed_length = fixed_length

    def process(self,
                parsed_sequence: Dict[str, Any]
                ) -> Dict[str, Any]:
        sequence = parsed_sequence[ITEM_SEQ_ENTRY_NAME]
        sequence = sequence[- self.fixed_length:]
        parsed_sequence[ITEM_SEQ_ENTRY_NAME] = sequence

        return parsed_sequence
