import io
import sys
from pathlib import Path

from asme.data.datasets import INT_BYTE_SIZE
from asme.data.multi_processing import MultiProcessSupport


class SequencePositionIndex(MultiProcessSupport):

    """
    This is a reader for a index file that contains a list of pairs containing the sequence id and a position
    of the sequence. A dataset using this reader can use the position e.g. to truncate the sequence.

    The index can be used, in the following scenarios:

    - you want to train your model on / evaluate your model for all positions in the sequence: e.g.
    consider the sequence [1, 2, 3, 4, 5] with id 42, the index can contain the following information:
    (42, 1), (42, 2), (42, 3) (note: positions start at 0) and the corresponding dataset can use
    the index to return the following sequences:
    [1], [1, 2], [1, 2, 3], [1, 2, 3, 4] maybe with the targets 2, 3, 4, 5 for training
    - for leave one out evaluation/training: one single sequence datafile can be used for the train, valid, and test
    using different sequence position indices
    """

    def __init__(self,
                 index_path: Path
                 ):
        """
        inits the sequence position index
        :param index_path:
        """
        super().__init__()
        if not index_path.exists():
            raise Exception(f"could not find file with index at: {index_path}")
        self._index_path = index_path
        self._init()

    def _init(self):
        with self._index_path.open("rb") as file_handle:
            self._length = self._read_length(file_handle)

    def _read_length(self, file_handle):
        file_handle.seek(-1 * INT_BYTE_SIZE, io.SEEK_END)
        return int.from_bytes(file_handle.read(INT_BYTE_SIZE), byteorder=sys.byteorder, signed=False)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        with self._index_path.open("rb") as file_handle:
            file_handle.seek(idx * INT_BYTE_SIZE * 2)
            session_idx = int.from_bytes(file_handle.read(INT_BYTE_SIZE), byteorder=sys.byteorder, signed=False)
            target_pos = int.from_bytes(file_handle.read(INT_BYTE_SIZE), byteorder=sys.byteorder, signed=False)

        return session_idx, target_pos

    def _init_class_for_worker(self, worker_id: int, num_worker: int, seed: int):
        self._init()
