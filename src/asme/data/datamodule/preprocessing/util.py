from pathlib import Path
from typing import List

from asme.data.base.reader import CsvDatasetIndex, CsvDatasetReader
from asme.data.datasets.sequence import PlainSequenceDataset, ItemSessionParser, MetaInformation
from asme.data.utils.csv import create_indexed_header, read_csv_header


def format_prefix(prefixes: List[str]) -> str:
    return ".".join(prefixes)


def create_session_data_set(column: MetaInformation,
                            data_file_path: Path,
                            index_file_path: Path,
                            delimiter: str) -> PlainSequenceDataset:
    """
    Helper method which returns a PlainSessionDataset for a given data and index file

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

    return PlainSequenceDataset(reader, parser)