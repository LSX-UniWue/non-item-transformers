from pytorch_lightning import seed_everything

from asme.data.datasets.processors.cloze_mask import ClozeMaskProcessor
from util_test import assert_list_equal
from util_test_tokenizer import create_tokenizer
from asme.data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME


def test_cloze_mask_processor_last_item():
    seed_everything(42)
    tokenizer = create_tokenizer()
    processor = ClozeMaskProcessor({"tokenizers.item": tokenizer}, 1.0, 1.0)

    parsed_session = {
        ITEM_SEQ_ENTRY_NAME: [5, 8, 9, 7, 3, 4]
    }
    masked = processor.process(parsed_session)

    assert_list_equal(masked[ITEM_SEQ_ENTRY_NAME], [5, 8, 9, 7, 3, tokenizer.mask_token_id])
    expected_targets = [tokenizer.pad_token_id] * 5 + [4]
    assert_list_equal(masked[TARGET_ENTRY_NAME], expected_targets)


def test_cloze_mask_processor():
    seed_everything(42)
    tokenizer = create_tokenizer()

    processor = ClozeMaskProcessor({"tokenizers.item": tokenizer}, 0.5, 0.1)

    parsed_session = {
        ITEM_SEQ_ENTRY_NAME: [5, 8, 9, 7, 3, 4, 12, 10, 11, 3]
    }
    masked = processor.process(parsed_session)

    assert_list_equal(masked[ITEM_SEQ_ENTRY_NAME], [5, 1, 9, 1, 3, 1, 12, 10, 1, 3])
    expected_targets = [0, 8, 0, 7, 0, 4, 0, 0, 11, 0]
    assert_list_equal(masked[TARGET_ENTRY_NAME], expected_targets)
