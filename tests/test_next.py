from pytorch_lightning import seed_everything
from torch.utils.data import RandomSampler

from util_test import assert_list_equal
from util_test_templating import load_dataset, get_all_data, assert_loo_test_and_validation, \
    assert_next_item_test_validation
from util_test_tokenizer import TEST_DATASET_BASE_PATH


LEAVE_ONE_OUT_TEMPLATE = {
    "template": {
        'name': 'next_sequence_step',
        'batch_size': 2,
        'num_workers': 0,
        'path': TEST_DATASET_BASE_PATH,
        'file_prefix': 'example',
        'split': 'leave_one_out'
    }
}

RATIO_TEMPLATE = {
    "template": {
        'name': 'next_sequence_step',
        'batch_size': 2,
        'num_workers': 0,
        'path': TEST_DATASET_BASE_PATH / 'ratio-0.8_0.1_0.1',
        'file_prefix': 'example',
        'split': 'ratio_split'
    }
}


def test_leave_one_out_next_template():
    seed_everything(42)

    data_sources = load_dataset(LEAVE_ONE_OUT_TEMPLATE)
    train_dataloader = data_sources['train']

    # check if train data loader is shuffled
    assert isinstance(train_dataloader.sampler, RandomSampler)

    train_data = get_all_data(train_dataloader)
    assert len(train_data) == 12  # 8 sequence have total length 4 and 2 sequences has total length 5

    # test some train data
    sequence, pos = train_data['2_2']
    assert_list_equal(sequence, [9, 10])
    assert pos == 11

    sequence, pos = train_data['2_1']
    assert_list_equal(sequence, [9])
    assert pos == 10

    seq2, pos2 = train_data['4_1']
    assert_list_equal(seq2, [7, 0])
    assert pos2 == 8

    assert_loo_test_and_validation(data_sources)


def test_ratio_next_template():
    seed_everything(42)
    data_sources = load_dataset(RATIO_TEMPLATE)

    train_dataloader = data_sources['train']
    assert isinstance(train_dataloader.sampler, RandomSampler)

    train_data = get_all_data(train_dataloader)
    assert len(train_data) == 26

    sequence, pos = train_data['0_1']
    assert_list_equal(sequence, [3, 0])
    assert pos == 10

    seq2, pos2 = train_data['3_4']
    assert_list_equal(seq2, [9, 10, 11, 12])
    assert pos2 == 7

    assert_next_item_test_validation(data_sources)
