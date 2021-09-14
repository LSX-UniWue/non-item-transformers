import random
from typing import List, Any, Dict, Optional

from torch.utils.data import Dataset

from asme.data.datasets import SAMPLE_IDS, ITEM_SEQ_ENTRY_NAME
from asme.data.datasets.index import SequencePositionIndex
from asme.data.datasets.processors.processor import Processor
from asme.data.datasets.sequence import PlainSequenceDataset
from asme.data.example_logging import ExampleLogger, Example
from asme.data.multi_processing import MultiProcessSupport


class SequencePositionDataset(Dataset, MultiProcessSupport, ExampleLogger):

    """
    A dataset that uses a sequence position index to load the session till the specified position in the index

    e.g. if the csv contains the sequence [9, 5, 6, 7] and the position 2 in the position index the sequence
    [9, 5, 6] is returned

    this is also applied to all other meta data that was parsed as a sequence
    """

    def __init__(self,
                 dataset: PlainSequenceDataset,
                 index: SequencePositionIndex,
                 sequences_to_truncate: Optional[List[str]] = None,
                 processors: Optional[List[Processor]] = None
                 ):
        super().__init__()
        self._dataset = dataset
        self._index = index
        if sequences_to_truncate is None:
            sequences_to_truncate = [ITEM_SEQ_ENTRY_NAME]
        self.sequences_to_truncate = sequences_to_truncate
        if processors is None:
            processors = []
        self._processors = processors

    def __len__(self):
        return len(self._index)

    def _get_example(self, idx: int) -> Example:
        sequence_data = self._get_raw_sequence(idx)
        processed_data = self._process_sequence(sequence_data.copy())

        return Example(sequence_data, processed_data)

    def _get_raw_sequence(self, idx: int) -> Dict[str, Any]:
        sequence_idx, position = self._index[idx]
        parsed_session = self._dataset[sequence_idx]
        parsed_session[SAMPLE_IDS] = sequence_idx
        parsed_session['pos'] = position

        for sequence_to_truncate in self.sequences_to_truncate:
            parsed_session[sequence_to_truncate] = parsed_session[sequence_to_truncate][:position + 1]
        return parsed_session

    def _process_sequence(self,
                          parsed_sequence: Dict[str, Any]
                          ) -> Dict[str, Any]:
        for processor in self._processors:
            parsed_sequence = processor.process(parsed_sequence)

        return parsed_sequence

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self._get_example(idx)
        return example.processed_data

    def _init_class_for_worker(self, worker_id: int, num_worker: int, seed: int):
        # nothing to do here
        pass

    def get_data_examples(self, num_examples: int = 1) -> List[Example]:
        return [
            self._get_example(example_id) for example_id in random.sample(range(0, len(self)), num_examples)
        ]
