import datetime
import functools
import io
import random
from typing import Dict, List, Any, Callable, Optional
import csv

from dataclasses import dataclass, field

from asme.tokenization.tokenizer import Tokenizer
from torch.utils.data import Dataset

from data.base.reader import CsvDatasetReader
from data.datasets import SAMPLE_IDS
from data.datasets.processors.processor import Processor
from data.example_logging import ExampleLogger, Example
from data.multi_processing import MultiProcessSupport


def _parse_boolean(text: str
                   ) -> bool:
    return text == 'True'


def _parse_timestamp(text: str,
                     date_format: str
                     ) -> datetime.datetime:
    return datetime.datetime.strptime(text, date_format)


def _parse_strlist(text: str,
                   delimiter: str) -> List[str]:
    return text.split(sep=delimiter)


def _identity(text: str
              ) -> str:
    return text


@dataclass
class MetaInformation:

    feature_name: str
    type: str
    tokenizer: Optional[Tokenizer] = None
    is_sequence: bool = True
    sequence_length: Optional[int] = None
    run_tokenization: bool = True  #TODO (AD) maybe think about finding a more user friendly default?
    is_generated: bool = False  # True iff the feature will be generated based on other features
    column_name: Optional[str] = None
    configs: Dict[str, Any] = field(default_factory=dict)

    def get_config(self, config_key: str) -> Optional[Any]:
        return self.configs.get(config_key, None)



# TODO: move to provider utils?
def _build_converter(info: MetaInformation
                     ) -> Callable[[str], Any]:
    feature_type = info.type
    if feature_type == 'int':
        return int

    if feature_type == 'str':
        return _identity

    if feature_type == 'bool':
        return _parse_boolean

    if feature_type == 'timestamp':
        return functools.partial(_parse_timestamp, date_format=info.get_config('format'))

    # FIXME: replace with a generic list convert that also converts the entries in the list
    if feature_type == 'strlist':
        return functools.partial(_parse_strlist, delimiter=info.get_config('delimiter'))

    raise KeyError(f'{feature_type} not supported. Currently only bool, timestamp and int are supported. '
                   f'See documentation for more details')


class SequenceParser:
    def parse(self, raw_session: str) -> Dict[str, Any]:
        raise NotImplementedError()


class ItemSessionParser(SequenceParser):

    def __init__(self,
                 indexed_headers: Dict[str, int],
                 features: List[MetaInformation],
                 delimiter: str = "\t"
                 ):
        super().__init__()
        self._indexed_headers = indexed_headers
        self._features = features
        self._delimiter = delimiter

    def parse(self,
              raw_session: str
              ) -> Dict[str, Any]:
        reader = csv.reader(io.StringIO(raw_session), delimiter=self._delimiter)
        entries = list(reader)
        parsed_data = {}
        for meta_info in self._features:
            feature_sequence = meta_info.is_sequence
            name = meta_info.column_name
            feature_key = meta_info.feature_name
            feature_column_name = name if name else feature_key
            # if feature changes over the sequence parse it over all entries, else extract it form the first entry
            if feature_sequence:
                feature = [self._get_feature(entry, feature_column_name, meta_info) for entry in entries]
            else:
                feature = self._get_feature(entries[0], feature_column_name, meta_info)
            parsed_data[feature_key] = feature

        return parsed_data

    def _get_feature(self,
                     entry: List[str],
                     feature_key: str,
                     info: MetaInformation
                     ) -> Any:
        converter = _build_converter(info)
        feature_idx = self._indexed_headers[feature_key]
        return converter(entry[feature_idx])


class PlainSequenceDataset(Dataset, MultiProcessSupport):
    """
    A dataset implementation that uses the CSVDatasetReader
    and the a SequenceParser

    """

    def __init__(self,
                 reader: CsvDatasetReader,
                 parser: SequenceParser
                 ):
        super().__init__()
        self._reader = reader
        self._parser = parser

    def __getitem__(self, idx):
        session = self._reader.get_sequence(idx)
        parsed_sequence = self._parser.parse(session)

        parsed_sequence[SAMPLE_IDS] = idx

        return parsed_sequence

    def __len__(self):
        return len(self._reader)

    def _init_class_for_worker(self, worker_id: int, num_worker: int, seed: int):
        # nothing to do here
        pass


class ItemSequenceDataset(Dataset, MultiProcessSupport, ExampleLogger):

    def __init__(self,
                 plain_sequence_dataset: PlainSequenceDataset,
                 processors: List[Processor] = None
                 ):
        super().__init__()
        self._plain_sequence_dataset = plain_sequence_dataset
        if processors is None:
            processors = []
        self._processors = processors

    def __len__(self):
        return len(self._plain_sequence_dataset)

    def __getitem__(self, idx):
        example = self._get_example(idx)
        return example.processed_data

    def _init_class_for_worker(self, worker_id: int, num_worker: int, seed: int):
        # nothing to do here
        pass

    def _get_raw_sequence(self, idx: int) -> Dict[str, Any]:
        return self._plain_sequence_dataset[idx]

    def _process_sequence(self,
                          parsed_sequence: Dict[str, Any]
                          ) -> Dict[str, Any]:
        for processor in self._processors:
            parsed_sequence = processor.process(parsed_sequence)

        return parsed_sequence

    def _get_example(self, idx: int) -> Example:
        sequence_data = self._get_raw_sequence(idx)
        processed_data = self._process_sequence(sequence_data.copy())

        return Example(sequence_data, processed_data)

    def get_data_examples(self, num_examples: int = 1) -> List[Example]:
        return [
            self._get_example(example_id) for example_id in random.sample(range(0, len(self)), num_examples)
        ]
