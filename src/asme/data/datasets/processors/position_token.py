from typing import List, Dict, Any

from asme.data.datasets import ITEM_SEQ_ENTRY_NAME, POSITION_IDS
from asme.data.datasets.processors.processor import Processor


class PositionTokenProcessor(Processor):
    """
    Processor that flattens the basket sequence and adds position ids for the flatten basket items
    """

    def __init__(self,
                 seq_length: int
                 ):
        super().__init__()

        self._seq_length = seq_length

    def _generate_position_tokens(self,
                                  items: List[int]
                                  ) -> List[int]:
        counts = list(map(len, items))

        positions = []
        last_position = 0
        for position, count in enumerate(counts):
            total_count = [position] * count
            positions.extend(total_count)
            last_position += len(total_count)
        # maybe to many items
        positions = positions[0: self._seq_length]
        # fill up the last positions
        end = last_position + self._seq_length - len(positions)
        positions.extend(range(last_position, end))

        assert len(positions) == self._seq_length
        return positions

    def process(self,
                parsed_sequence: Dict[str, Any]
                ) -> Dict[str, Any]:
        items = parsed_sequence[ITEM_SEQ_ENTRY_NAME]
        if not isinstance(items[0], list):
            raise ValueError('sequence items are not list of lists')

        # generate the positions
        positions = self._generate_position_tokens(items)

        flat_items = [item for sublist in items for item in sublist]
        parsed_sequence[ITEM_SEQ_ENTRY_NAME] = flat_items
        parsed_sequence[POSITION_IDS] = positions

        return parsed_sequence
