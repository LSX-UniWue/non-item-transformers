from typing import Dict, Any

from data.datasets.sequence import MetaInformation
from torch.utils.data import DataLoader

from asme.init.config import Config
from asme.init.context import Context
from asme.init.factories.data_sources.template_datasources import TemplateDatasourceFactory
from asme.init.templating.template_engine import TemplateEngine
from data.datasets import SAMPLE_IDS, ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from util_test import assert_list_equal
from util_test_tokenizer import create_tokenizer


"""
the number of sequences in the example csv file
"""
NUM_EXAMPLES_SEQUENCES = 10


def assert_loo_test_and_validation(data_sources: Dict[str, DataLoader]):
    # test some test data
    test_dataloader = data_sources['test']
    test_data = get_all_data(test_dataloader)

    assert len(test_data) == NUM_EXAMPLES_SEQUENCES

    test_sequence, test_target = test_data['2_4']
    assert_list_equal(test_sequence, [9, 10, 11, 12])
    assert test_target == 7

    # test some validation data
    val_dataloader = data_sources['validation']
    val_data = get_all_data(val_dataloader)

    assert len(val_data) == NUM_EXAMPLES_SEQUENCES

    val_sequence, val_target = val_data['5_2']
    assert_list_equal(val_sequence, [4, 7])
    assert val_target == 10


def assert_next_item_test_validation(data_sources: Dict[str, DataLoader]):
    test_dataloader = data_sources['test']
    test_data = get_all_data(test_dataloader)
    assert len(test_data) == 3

    test_seq, test_target = test_data['0_1']
    assert_list_equal(test_seq, [5, 0])
    assert test_target == 6

    test_seq2, test_target2 = test_data['0_2']
    assert_list_equal(test_seq2, [5, 6])
    assert test_target2 == 8

    val_dataloader = data_sources['validation']
    val_data = get_all_data(val_dataloader)

    assert len(val_data) == 3

    val_seq, val_target = val_data['0_1']
    assert_list_equal(val_seq, [3, 0])
    assert val_target == 4

    val_seq2, val_target2 = val_data['0_3']
    assert_list_equal(val_seq2, [3, 4, 7])
    assert val_target2 == 8


def load_dataset(template: Dict[str, Any]
                 ) -> Dict[str, DataLoader]:
    modified_template = TemplateEngine().modify(template)
    print(modified_template)
    context = Context()

    item_seq_feature_info = MetaInformation('item', type='str', column_name="item_id", is_sequence=True,
                                            sequence_length=4)

    context.set('tokenizers.item', create_tokenizer())
    context.set('features', [item_seq_feature_info])

    config = Config(modified_template.get('data_sources'))
    return TemplateDatasourceFactory().build(config, context)


def get_all_data(data_loader: DataLoader
                  ) -> Dict[str, Any]:
    data = {}
    for batch in data_loader:
        data_ids = batch[SAMPLE_IDS].tolist()
        positions = batch['pos'].tolist() if 'pos' in batch else [0] * len(data_ids)
        sequences = batch[ITEM_SEQ_ENTRY_NAME].tolist()
        targets = batch[TARGET_ENTRY_NAME].tolist()

        for data_id, pos, sequence, target in zip(data_ids, positions, sequences, targets):
            data[f'{data_id}_{pos}'] = sequence, target

    return data
