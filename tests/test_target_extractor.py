from asme.data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from asme.data.datasets.processors.target_extractor import TargetExtractorProcessor
from util_test import assert_list_equal


def test_target_extractor_processor():
    target_extractor_processor = TargetExtractorProcessor()

    parsed_sequence = {
        ITEM_SEQ_ENTRY_NAME: [5, 8, 9, 7, 3, 4]
    }

    sequence = target_extractor_processor.process(parsed_sequence)

    assert_list_equal(sequence[ITEM_SEQ_ENTRY_NAME], [5, 8, 9, 7, 3])
    assert sequence[TARGET_ENTRY_NAME] == 4
