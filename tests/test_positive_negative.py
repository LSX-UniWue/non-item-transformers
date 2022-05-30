from typing import Any, Dict

from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, RandomSampler

from util_test import assert_list_equal
from util_test_templating import load_dataset, NUM_EXAMPLES_SEQUENCES, assert_loo_test_and_validation, \
    assert_next_item_test_validation
from util_test_tokenizer import TEST_DATASET_BASE_PATH
from asme.data.datasets import SAMPLE_IDS, ITEM_SEQ_ENTRY_NAME, POSITIVE_SAMPLES_ENTRY_NAME, NEGATIVE_SAMPLES_ENTRY_NAME


LEAVE_ONE_OUT_TEMPLATE = {
    "template": {
        'name': 'pos_neg',
        'batch_size': 2,
        'num_workers': 0,
        'path': TEST_DATASET_BASE_PATH,
        'file_prefix': 'example',
        'split': 'leave_one_out'
    }
}

RATIO_TEMPLATE = {
    "template": {
        'name': 'pos_neg',
        'batch_size': 2,
        'num_workers': 0,
        'path': TEST_DATASET_BASE_PATH / 'ratio-0.8_0.1_0.1',
        'file_prefix': 'example',
        'split': 'ratio_split'
    }
}


def _get_all_data_positive_negative(data_loader: DataLoader
                                    ) -> Dict[str, Any]:
    data = {}
    for batch in data_loader:
        data_ids = batch[SAMPLE_IDS].tolist()
        positions = batch['pos'].tolist() if 'pos' in batch else [0] * len(data_ids)
        sequences = batch[ITEM_SEQ_ENTRY_NAME].tolist()
        positive_examples = batch[POSITIVE_SAMPLES_ENTRY_NAME].tolist()
        negative_examples = batch[NEGATIVE_SAMPLES_ENTRY_NAME].tolist()

        for data_id, pos, sequence, positive_example, negative_example in zip(data_ids, positions, sequences,
                                                                              positive_examples, negative_examples):
            data[f'{data_id}_{pos}'] = sequence, positive_example, negative_example

    return data


def test_leave_one_out_positive_negative_template():
    seed_everything(42)

    data_sources = load_dataset(LEAVE_ONE_OUT_TEMPLATE)
    train_dataloader = data_sources['train']

    # check if train data loader is shuffled
    assert isinstance(train_dataloader.sampler, RandomSampler)

    train_data = _get_all_data_positive_negative(train_dataloader)
    assert len(train_data) == NUM_EXAMPLES_SEQUENCES

    # test some train data
    sequence, pos, neg = train_data['0_1']
    assert_list_equal(sequence, [3])
    assert_list_equal(pos, [4])
    assert_list_equal(neg, [5])
    seq2, pos2, neg2 = train_data['4_1']
    assert_list_equal(seq2, [7])
    assert_list_equal(pos2, [8])
    assert_list_equal(neg2, [12])

    assert_loo_test_and_validation(data_sources)


def test_ratio_positive_negative_template():
    seed_everything(42)
    data_sources = load_dataset(RATIO_TEMPLATE)

    train_dataloader = data_sources['train']
    assert isinstance(train_dataloader.sampler, RandomSampler)

    train_data = _get_all_data_positive_negative(train_dataloader)
    assert len(train_data) == 8

    sequence, pos, neg = train_data['0_0']
    assert_list_equal(sequence, [3, 10, 6])
    assert_list_equal(pos, [10, 6, 5])
    assert_list_equal(neg, [4, 4, 9])

    seq2, pos2, neg2 = train_data['4_0']
    assert_list_equal(seq2, [7, 8, 3, 0])
    assert_list_equal(pos2, [8, 3, 6, 0])
    assert_list_equal(neg2, [9, 4, 11, 0])  # TODO: check and discuss

    assert_next_item_test_validation(data_sources)
