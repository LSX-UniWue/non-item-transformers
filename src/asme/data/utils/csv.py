from pathlib import Path
from typing import List, Dict

import csv


def read_csv_header(data_file_path: Path, delimiter: str) -> List[str]:
    """
    Parses the first line of the file as csv.

    :param data_file_path: path to a csv file.
    :param delimiter: the delimiter used during csv encoding

    :return: a list with column names
    """
    with data_file_path.open("r") as file:
        reader = csv.reader(file, delimiter=delimiter)
        return next(reader)


def create_indexed_header(header: List[str]) -> Dict[str, int]:
    """
    Creates a mapping from the column names, to their column index.

    :param header: a list of csv headers
    :return: a mapping from column name to column index.
    """
    return {
        name: idx for idx, name in enumerate(header)
    }
