import functools
from operator import itemgetter
from pathlib import Path
from typing import Dict, Any, Iterable, Callable, Optional, List
from asme.data.datasets import ITEM_SEQ_ENTRY_NAME
from asme.data.base.reader import CsvDatasetIndex, CsvDatasetReader
from asme.data.datasets.index_builder import SequencePositionIndexBuilder
from asme.data.datasets.sequence import ItemSequenceDataset, ItemSessionParser, PlainSequenceDataset, MetaInformation
from asme.data.utils.csv import create_indexed_header, read_csv_header
from asme.datasets.data_structures.dataset_metadata import DatasetMetadata


def all_remaining_positions(session: Dict[str, Any]
                            ) -> Iterable[int]:
    return range(1, len(session[ITEM_SEQ_ENTRY_NAME]) - 2)


def _get_position_with_offset(session: Dict[str, Any],
                              offset: int
                              ) -> Iterable[int]:
    """
    returns an iterator that only contains the position len(sequence) - offset
    :param session:
    :param offset:
    :return:
    """
    sequence = session[ITEM_SEQ_ENTRY_NAME]
    return [len(sequence) - offset]


def filter_by_sequence_feature(sequence: Dict[str, Any],
                               feature_key: str,
                               min_sequence_length: int
                               ) -> Iterable[int]:
    feature_values = sequence[feature_key]
    targets = list(filter(itemgetter(1), enumerate(feature_values)))
    target_idxs: List[int] = list(map(itemgetter(0), targets))

    for forbidden_position in range(0, min_sequence_length - 1):
        if forbidden_position in target_idxs:
            target_idxs.remove(forbidden_position)

    return target_idxs


def _build_target_position_extractor(target_feature: str,
                                     min_sequence_length: int
                                     ) -> Optional[Callable[[Dict[str, Any]], Iterable[int]]]:
    if target_feature is None:
        return None

    return functools.partial(filter_by_sequence_feature, feature_key=target_feature,
                             min_sequence_length=min_sequence_length)


def create_conditional_index(dataset_metadata: DatasetMetadata,
                             output_file_path: Path,
                             target_feature: Optional[str],
                             min_sequence_length: Optional[int] = None
                             ) -> None:
    """
    FixMe I need some documentation
    :param dataset_metadata: Data Set Metadata
    :param output_file_path: path to the output file
    :param target_feature: the target column name to build the targets against,
    (default all next subsequences will be considered); the target must be a boolean feature
    :param min_sequence_length: the minimum sequence length when extracting the target
    :return:
    """
    target_positions_extractor = _build_target_position_extractor(target_feature, min_sequence_length)
    additional_features = {}
    if target_feature is not None:
        additional_features[target_feature] = {'type': 'bool', 'sequence': True}

    create_conditional_index_using_extractor(dataset_metadata=dataset_metadata,
                                             output_file_path=output_file_path,
                                             target_positions_extractor=target_positions_extractor,
                                             additional_features=additional_features)


def create_conditional_index_using_extractor(dataset_metadata: DatasetMetadata,
                                             output_file_path: Path,
                                             target_positions_extractor: Callable[[Dict[str, Any]], Iterable[int]],
                                             additional_features: Optional[Dict[str, Any]] = None) -> None:
    """
    Create a new conditional Index from an already existing one via target position extractor.
    :param dataset_metadata: Data Set Metadata
    :param output_file_path: file that the index is stored to
    :param additional_features: FixMe I need a description
    :param target_positions_extractor: FixMe I need a description
    :return: None, Side effect: new index file is stored at output_file_path
    """
    if additional_features is None:
        additional_features = {}
    dataset = _build_item_sequence_dataset(dataset_metadata, additional_features)

    builder = SequencePositionIndexBuilder(target_positions_extractor=target_positions_extractor)
    builder.build(dataset, output_file_path)


def _build_item_sequence_dataset(dataset_metadata: DatasetMetadata,
                                 additional_features: Optional[Dict[str, Any]] = None
                                 ) -> ItemSequenceDataset:
    features = [MetaInformation('item', 'str', column_name=dataset_metadata.item_header_name)]
    if additional_features is not None:
        for feature_name, info in additional_features.items():
            feature_meta_data = MetaInformation(feature_name, 'bool')
            features.append(feature_meta_data)

    session_parser = ItemSessionParser(
        create_indexed_header(read_csv_header(dataset_metadata.data_file_path, dataset_metadata.delimiter)),
        features,
        delimiter=dataset_metadata.delimiter
    )

    session_index = CsvDatasetIndex(dataset_metadata.session_index_path)
    reader = CsvDatasetReader(dataset_metadata.data_file_path, session_index)

    plain_dataset = PlainSequenceDataset(reader, session_parser)
    return ItemSequenceDataset(plain_dataset)


def _window_position_generator(data: Dict[str, Any],
                               window_size: int,
                               session_end_offset: int = 0
                               ) -> Iterable[int]:
    sequence = data[ITEM_SEQ_ENTRY_NAME]
    return range(window_size - 1, len(sequence) - session_end_offset)


def create_sliding_window_index(dataset_metadata: DatasetMetadata,
                                output_file_path: Path,
                                window_size: int,
                                session_end_offset: int = 0
                                ) -> None:
    dataset = _build_item_sequence_dataset(dataset_metadata)

    builder = SequencePositionIndexBuilder(functools.partial(_window_position_generator, window_size=window_size,
                                                             session_end_offset=session_end_offset))
    builder.build(dataset, output_file_path)


def run_loo_split(dataset_metadata: DatasetMetadata,
                  output_dir_path: Path):
    """
    Creates a next item split, i.e., From every session with length k use the sequence[0:k-2] for training,
    sequence[-2] for validation and sequence[-1] for testing.

    :param dataset_metadata: Data Set Metadata
    :param output_dir_path: output directory where the index files for the splits are written to
    :return:
    """
    # Create train index, cut the sequences at k - 3
    create_conditional_index_using_extractor(
        dataset_metadata=dataset_metadata,
        output_file_path=output_dir_path / (dataset_metadata.file_prefix + ".train.loo.idx"),
        target_positions_extractor=functools.partial(_get_position_with_offset, offset=3)
    )

    # Create validation index with target item n-1
    create_conditional_index_using_extractor(dataset_metadata=dataset_metadata, output_file_path=output_dir_path / (
                dataset_metadata.file_prefix + ".validation.loo.idx"),
                                             target_positions_extractor=functools.partial(_get_position_with_offset, offset=2))
    # Create testing index with target item n
    create_conditional_index_using_extractor(dataset_metadata=dataset_metadata, output_file_path=output_dir_path / (
                dataset_metadata.file_prefix + ".test.loo.idx"),
                                             target_positions_extractor=functools.partial(_get_position_with_offset, offset=1))
