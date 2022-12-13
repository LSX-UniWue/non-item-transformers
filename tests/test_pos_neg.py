from pytorch_lightning import seed_everything

from asme.data.datasets import ITEM_SEQ_ENTRY_NAME, POSITIVE_SAMPLES_ENTRY_NAME, NEGATIVE_SAMPLES_ENTRY_NAME
from asme.data.datasets.processors.pos_neg_sampler import PositiveNegativeSamplerProcessor
from util_test import assert_list_equal
from util_test_tokenizer import create_tokenizer


def test_positive_negative_sampler_processor():
    seed_everything(42)
    tokenizers = {}
    tokenizers[ITEM_SEQ_ENTRY_NAME] = create_tokenizer()
    pos_neg_sampler = PositiveNegativeSamplerProcessor(tokenizers)

    parsed_sequence = {
        ITEM_SEQ_ENTRY_NAME: [5, 8, 9, 7, 3, 4]
    }

    sequence = pos_neg_sampler.process(parsed_sequence)

    assert_list_equal(sequence[ITEM_SEQ_ENTRY_NAME], [5, 8, 9, 7, 3])
    assert_list_equal(sequence[POSITIVE_SAMPLES_ENTRY_NAME], [8, 9, 7, 3, 4])
    assert_list_equal(sequence[NEGATIVE_SAMPLES_ENTRY_NAME], [6, 6, 6, 6, 11])


def test_positive_negative_sampler_processor_basket_recommendation():
    seed_everything(42)
    tokenizers = {}
    tokenizers[ITEM_SEQ_ENTRY_NAME] = create_tokenizer()
    pos_neg_sampler = PositiveNegativeSamplerProcessor(tokenizers)

    parsed_sequence = {
        ITEM_SEQ_ENTRY_NAME: [[5, 3], [8], [9], [7, 5], [3], [4]]
    }

    sequence = pos_neg_sampler.process(parsed_sequence)

    assert_list_equal(sequence[ITEM_SEQ_ENTRY_NAME], [[5, 3], [8], [9], [7, 5], [3]])
    assert_list_equal(sequence[POSITIVE_SAMPLES_ENTRY_NAME], [[8], [9], [7, 5], [3], [4]])
    assert_list_equal(sequence[NEGATIVE_SAMPLES_ENTRY_NAME], [[6, 6], [6], [6], [11, 10], [12]])
