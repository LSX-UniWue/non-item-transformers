from pathlib import Path
from typing import List, Union, Optional


class DatasetMetadata:
    def __init__(self, data_file_path: Path,
                 session_key: Optional[List[str]],
                 item_header_name: Union[int, str],
                 delimiter: str,
                 session_index_path: Optional[Path] = None,
                 special_tokens: Optional[List[str]] = None,
                 stats_columns: Optional[List[Union[str, int]]] = None):
        """
        :param data_file_path: CSV input file for the dataset in a delimiter separated file
        :param session_key: Session identifier name in data set header
        :param item_header_name: data set key that the item-ids are stored under
        :param special_tokens: special tokens that are to be included in the vocabulary
        :param stats_columns: columns that vocabulary and popularity are created for
        :param delimiter: Delimiter used in original data file
        """

        self.data_file_path: Path = data_file_path
        self.session_key = session_key
        self.item_header_name = item_header_name
        self.delimiter = delimiter
        self.file_prefix = data_file_path.stem
        self.dataset_base_dir: Path = data_file_path.parent
        # file path to session index
        if session_index_path is None:
            self.session_index_path = self.dataset_base_dir.joinpath(self.file_prefix + '.session.idx')
        else:
            self.session_index_path = session_index_path

        if session_key is None:
            self.session_key = ["No session key available"]
        else:
            self.session_key = session_key

        if special_tokens is None:
            self.custom_tokens: List[str] = []
        else:
            self.custom_tokens: List[str] = special_tokens

        if stats_columns is None:
            self.stats_columns: List[str] = [self.item_header_name]
        else:
            self.stats_columns: List[str] = [self.item_header_name] + stats_columns
