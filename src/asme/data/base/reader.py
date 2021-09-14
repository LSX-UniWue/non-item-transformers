from dataclasses import dataclass
from pathlib import Path
import sys
import io

from asme.data.multi_processing import MultiProcessSupport


@dataclass
class SequenceBoundary:
    """
    a dataclass representing the start and end of a sequence
    """
    start: int
    end: int


class CsvDatasetIndex(MultiProcessSupport):
    """
    Index to get the start position and end position of a sequence in a csv file containing sequences.
    """

    INT_BYTE_SIZE = 8

    def __init__(self,
                 index_file_path: Path
                 ):
        """
        inits the CsvDatasetIndex
        :param index_file_path: the path to the index file for a csv file containing sequences
        """
        self._index_file_path = index_file_path

        self._init()

    def _init(self):
        with self._index_file_path.open("rb") as file_handle:
            self._num_sequences = self._read_num_sequences(file_handle)

    def _read_num_sequences(self, file_handle) -> int:
        self._seek_to_header(file_handle)
        return self._read_int(file_handle)

    def _seek_to_header(self, file_handle):
        file_handle.seek(-self.INT_BYTE_SIZE, io.SEEK_END)

    def _read_int(self, file_handle) -> int:
        return int.from_bytes(file_handle.read(self.INT_BYTE_SIZE), byteorder=sys.byteorder, signed=False)

    def num_sequences(self) -> int:
        return self._num_sequences

    def __len__(self):
        return self.num_sequences()

    def get(self, session_num: int) -> SequenceBoundary:
        """
        Returns the boundaries of a specific session as byte positions within the file. The sessions are sequentially
        numbered and 0-based.

        :param session_num: a number in [0; num_sessions-1]
        :return: a tuple with start and end byte position within the data file.
        """
        with self._index_file_path.open("rb") as file_handle:
            file_handle.seek(session_num * self.INT_BYTE_SIZE * 2)

            start = self._read_int(file_handle)
            end = self._read_int(file_handle)

            return SequenceBoundary(start, end)

    def _init_class_for_worker(self, worker_id: int, num_worker: int, seed: int):
        self._init()


class CsvDatasetReader(MultiProcessSupport):

    """
    a dataset reader for the csv file
    """

    def __init__(self,
                 data_file_path: Path,
                 index: CsvDatasetIndex
                 ):
        """
        inits a CSVDatasetReader
        :param data_file_path: the path to the csv file containing the sequences
        :param index: the csv index to use for random access
        """
        self._data_file_path = data_file_path
        self._index = index

        self._init()

    def _init(self):
        self._num_sequences = self._index.num_sequences()

    def __len__(self):
        return self._num_sequences

    def get_sequence(self,
                     idx: int
                     ) -> str:
        """
        :param idx: the id of the sequence
        :return: the sequence with the specified idx in the index
        """
        if idx >= self._num_sequences:
            raise Exception(f"{idx} is not a valid index in [0, {self._num_sequences}]")

        sequence_boundary = self._index.get(idx)
        start = sequence_boundary.start
        end = sequence_boundary.end
        sequence_raw = self._read_sequence(start, end)

        return sequence_raw

    def _read_sequence(self, start: int, end: int) -> str:
        with self._data_file_path.open("rb") as file_handle:
            file_handle.seek(start)
            return file_handle.read(end - start).decode(encoding="utf-8")

    def _init_class_for_worker(self, worker_id: int, num_worker: int, seed: int):
        self._init()
