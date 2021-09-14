import sys
from pathlib import Path
from typing import Callable, Dict, Any, Iterable

from tqdm import tqdm

from asme.data.datasets import ITEM_SEQ_ENTRY_NAME, INT_BYTE_SIZE
from asme.data.datasets.sequence import ItemSequenceDataset


class SequencePositionIndexBuilder:
    """
    This index builder builds a sequence position index. For each sequence returned by an ItemSessionDataset,
    this builder can generate m different positions for the sequence.
    For example you will test every position of a sequence, you return a range from 1 to the sequence length.
    Than the sequence will be used sequence length - 1 times with different targets
    """

    def __init__(self, target_positions_extractor: Callable[[Dict[str, Any]], Iterable[int]] = None):
        if target_positions_extractor is None:
            target_positions_extractor = all_remaining_items
        self._target_positions_extractor = target_positions_extractor

    def build(self,
              dataset: ItemSequenceDataset,
              index_path: Path
              ):
        if not index_path.exists():
            index_path.parent.mkdir(parents=True, exist_ok=True)

        current_idx = 0
        with index_path.open("wb") as index_file:
            for session_idx in tqdm(range(len(dataset)), desc="Creating Index."):
                session = dataset[session_idx]

                # remove session with lower min session length
                target_positions = self._target_positions_extractor(session)
                for target_pos in target_positions:
                    # skip all session with target that do not satisfy the min session length

                    self._write_entry(index_file, session_idx, target_pos)
                    current_idx += 1
            # write length at the end
            index_file.write(current_idx.to_bytes(INT_BYTE_SIZE, byteorder=sys.byteorder, signed=False))

    @staticmethod
    def _write_entry(index_file, session_idx: int, target_pos: int):
        index_file.write(session_idx.to_bytes(INT_BYTE_SIZE, byteorder=sys.byteorder, signed=False))
        index_file.write(target_pos.to_bytes(INT_BYTE_SIZE, byteorder=sys.byteorder, signed=False))


def all_remaining_items(session: Dict[str, Any]
                        ) -> Iterable[int]:
    return range(1, len(session[ITEM_SEQ_ENTRY_NAME]))
