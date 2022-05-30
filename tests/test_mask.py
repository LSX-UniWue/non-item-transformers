from pytorch_lightning import seed_everything
from torch.utils.data import RandomSampler

from util_test import assert_list_equal
from util_test_templating import load_dataset, get_all_data, NUM_EXAMPLES_SEQUENCES
from util_test_tokenizer import TEST_DATASET_BASE_PATH

LEAVE_ONE_OUT_TEMPLATE = {
    "template": {
        'name': 'masked',
        'batch_size': 2,
        'num_workers': 0,
        'path': TEST_DATASET_BASE_PATH,
        'file_prefix': 'example',
        'mask_probability': 0.1,
        'mask_seed': 123456,
        'split': 'leave_one_out'
    }
}

RATIO_TEMPLATE = {
    "template": {
        'name': 'masked',
        'batch_size': 2,
        'num_workers': 0,
        'path': TEST_DATASET_BASE_PATH / 'ratio-0.8_0.1_0.1',
        'file_prefix': 'example',
        'mask_probability': 0.1,
        'mask_seed': 123456,
        'split': 'ratio_split'
    }
}


def test_leave_one_out_mask_template():
    seed_everything(42)

    data_sources = load_dataset(LEAVE_ONE_OUT_TEMPLATE)
    train_dataloader = data_sources['train']

    # check if train data loader is shuffled
    assert isinstance(train_dataloader.sampler, RandomSampler)

    train_data = get_all_data(train_dataloader)
    assert len(train_data) == NUM_EXAMPLES_SEQUENCES

    # test some train data
    sequence, target = train_data['0_1']
    assert_list_equal(sequence, [3, 4])
    assert_list_equal(target, [0, 0])
    seq2, target2 = train_data['4_1']
    assert_list_equal(seq2, [7, 8])
    assert_list_equal(target2, [0, 0])

    # test some test data
    test_dataloader = data_sources['test']
    test_data = get_all_data(test_dataloader)

    assert len(test_data) == NUM_EXAMPLES_SEQUENCES

    test_sequence, test_target = test_data['2_4']
    assert_list_equal(test_sequence, [10, 11, 12, 1])
    assert test_target == 7

    # test some validation data
    val_dataloader = data_sources['validation']
    val_data = get_all_data(val_dataloader)

    assert len(val_data) == NUM_EXAMPLES_SEQUENCES

    val_sequence, val_target = val_data['1_2']
    assert_list_equal(val_sequence, [3, 4, 1])
    assert val_target == 7


def test_ratio_mask_template():
    seed_everything(42)
    data_sources = load_dataset(RATIO_TEMPLATE)

    train_dataloader = data_sources['train']
    assert isinstance(train_dataloader.sampler, RandomSampler)

    train_data = get_all_data(train_dataloader)
    assert len(train_data) == 8

    sequence, target = train_data['0_0']
    assert_list_equal(sequence, [3, 10, 6, 5])
    assert_list_equal(target, [0, 0, 0, 0])

    seq2, target2 = train_data['4_0']
    assert_list_equal(seq2, [7, 8, 3, 1])
    assert_list_equal(target2, [0, 0, 0, 6])

    test_dataloader = data_sources['test']
    test_data = get_all_data(test_dataloader)

    assert len(test_data) == 3
    test_seq, test_target = test_data['0_1']
    assert_list_equal(test_seq, [5, 1, 0])
    assert test_target == 6

    test_seq2, test_target2 = test_data['0_2']
    assert_list_equal(test_seq2, [5, 6, 1])
    assert test_target2 == 8

    val_dataloader = data_sources['validation']
    val_data = get_all_data(val_dataloader)

    assert len(val_data) == 3

    val_seq, val_target = val_data['0_1']
    assert_list_equal(val_seq, [3, 1, 0])
    assert val_target == 4

    val_seq2, val_target2 = val_data['0_3']
    assert_list_equal(val_seq2, [3, 4, 7, 1])
    assert val_target2 == 8
