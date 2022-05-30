from typing import Union, List, Dict, Any

from asme.data.datasets import ITEM_SEQ_ENTRY_NAME, SAMPLE_IDS, POSITIVE_SAMPLES_ENTRY_NAME
from asme.data.datasets.processors.processor import Processor


class PositiveItemExtractorProcessor(Processor):

    """
    Takes the input sequence and generates positive items by taking the last {num_pos_items} items of the sequence.

    Example: num_pos_items=3
        Input:
            session: [1, 5, 7, 8, 10]
        Output:
            session:          [1, 5]
            positive samples: [7, 8, 10]

    """

    def __init__(self,
                 num_pos_items: int = 3
                 ):
        super().__init__()
        self.num_pos_items = num_pos_items

    def _extract_positive_items(self,
                                session: Union[List[int], List[List[int]]]
                                ) -> Union[List[int], List[List[int]]]:
        return session[-self.num_pos_items:]

    def process(self, parsed_session: Dict[str, Any]) -> Dict[str, Any]:
        session = parsed_session[ITEM_SEQ_ENTRY_NAME]

        if len(session) < (self.num_pos_items + 1):
            print(session)
            raise AssertionError(f'{parsed_session[SAMPLE_IDS]}')

        x = session[:-self.num_pos_items]

        pos = self._extract_positive_items(session)
        parsed_session[ITEM_SEQ_ENTRY_NAME] = x
        parsed_session[POSITIVE_SAMPLES_ENTRY_NAME] = pos

        return parsed_session
