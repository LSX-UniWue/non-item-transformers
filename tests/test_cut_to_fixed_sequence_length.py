from typing import Any, Dict

from asme.data.datasets import ITEM_SEQ_ENTRY_NAME
from asme.data.datasets.processors.cut_to_fixed_sequence_length import CutToFixedSequenceLengthProcessor
from pytorch_lightning import seed_everything

from util_test import assert_list_equal


def _build_sequence() -> Dict[str, Any]:
    return {
        ITEM_SEQ_ENTRY_NAME: [5, 8, 9, 7, 3, 4]
    }


def test_cut_to_fixed_sequence_length_processor():
    seed_everything(42)
    processor = CutToFixedSequenceLengthProcessor(3)

    parsed_session = _build_sequence()
    cut = processor.process(parsed_session)

    assert_list_equal(cut[ITEM_SEQ_ENTRY_NAME], [7, 3, 4])

    processor = CutToFixedSequenceLengthProcessor(5)

    parsed_session = _build_sequence()
    cut2 = processor.process(parsed_session)

    assert_list_equal(cut2[ITEM_SEQ_ENTRY_NAME], [8, 9, 7, 3, 4])
