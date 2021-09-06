import os
from pathlib import Path
from typing import List, Callable, Any, Optional
from tqdm import tqdm

from data.base.reader import CsvDatasetIndex, CsvDatasetReader
from data.datasets import ITEM_SEQ_ENTRY_NAME
from data.datasets.sequence import PlainSequenceDataset, ItemSessionParser, MetaInformation
from data.utils.csv import create_indexed_header, read_csv_header
from asme.tokenization.vocabulary import VocabularyBuilder, CSVVocabularyReaderWriter


def create_session_data_set(column: MetaInformation,
                            data_file_path: Path,
                            index_file_path: Path,
                            delimiter: str) -> PlainSequenceDataset:
    """
    Helper method wich returns a PlainSessionDataset for a given data and index file

    :param item_header_name: Name of the item key in the data set, e.g, "ItemId"
    :param data_file_path: Path to CSV file containing original data
    :param index_file_path: Path to index file belonging to the data file
    :param delimiter: delimiter used in data file
    :return: PlainSessionDataset
    """
    reader_index = CsvDatasetIndex(index_file_path)
    reader = CsvDatasetReader(data_file_path, reader_index)
    parser = ItemSessionParser(create_indexed_header(read_csv_header(data_file_path, delimiter=delimiter)),
                               [column],
                               delimiter=delimiter)

    session_data_set = PlainSequenceDataset(reader, parser)
    return session_data_set


def create_token_vocabulary(column: MetaInformation,
                            data_file_path: Path,
                            session_index_path: Path,
                            vocabulary_output_file_path: Path,
                            custom_tokens: List[str],
                            delimiter: str,
                            strategy_function: Optional[Callable[[List[Any]], List[Any]]] = None):
    """
    Creates a token vocabulary for the items in the data set in data file path.

    :param data_file_path: Path to CSV file containing original data
    :param session_index_path: Path to index file belonging to the data file
    :param vocabulary_output_file_path: output path for vocabulary file
    :param column: Name of the item key in the data set, e.g, "ItemId" TODO
    :param custom_tokens: FixMe I need documentation
    :param delimiter: delimiter used in data file
    :param strategy_function: function selecting which items of a session are used in the vocabulary
    :return: None, Side Effect: vocabulary for data file is written to vocabulary_output_file_path
    """
    vocab_builder = VocabularyBuilder()
    for token in custom_tokens:
        vocab_builder.add_token(token)

    sub_delimiter = column.get_config("delimiter")

    data_set = create_session_data_set(column=column,
                                       data_file_path=data_file_path,
                                       index_file_path=session_index_path,
                                       delimiter=delimiter)
    if strategy_function is None:
        def strategy_function(x: List[Any]) -> List[Any]:
            return x

    feature_name = column.feature_name if column.feature_name else column.column_name

    for idx in tqdm(range(len(data_set)), desc=f"Tokenizing items from: {data_file_path}"):
        session = data_set[idx]
        session_tokens = strategy_function(session[feature_name])

        for token in session_tokens:
            if sub_delimiter is not None:
                token = token.split(sub_delimiter)
                for word in token:
                    vocab_builder.add_token(word)
            else:
                vocab_builder.add_token(token)

    vocabulary = vocab_builder.build()

    if not os.path.exists(vocabulary_output_file_path.parent):
        vocabulary_output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with vocabulary_output_file_path.open("w") as file:
        CSVVocabularyReaderWriter().write(vocabulary, file)
