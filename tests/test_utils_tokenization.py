from collections import OrderedDict

from asme.core.tokenization.tokenizer import Tokenizer
from asme.core.tokenization.utils.tokenization import remove_special_tokens
from asme.core.tokenization.vocabulary import Vocabulary

from util_test import assert_list_equal


def tokenizer_with_special_tokens() -> Tokenizer:
    sample_vocabulary = OrderedDict([("[UNK]", 0), ("[PAD]", 0), ("[MASK]", 0), ("a", 3), ("b", 4), ("c", 5)])
    vocabulary = Vocabulary(sample_vocabulary)

    return Tokenizer(vocabulary, unk_token="[UNK]", pad_token="[PAD]", mask_token="[MASK]")


def test_remove_special_tokens():
    tokenizer = tokenizer_with_special_tokens()
    clean_sequence = ['a', 'b', 'c']
    sequence = tokenizer.convert_tokens_to_ids(clean_sequence + ['[PAD]', '[MASK]'])
    removed_seq = remove_special_tokens(sequence, tokenizer)

    assert_list_equal(removed_seq, tokenizer.convert_tokens_to_ids(clean_sequence))


def test_remove_special_tokens_basket():
    tokenizer = tokenizer_with_special_tokens()
    sequence = [['a', 'b'], ['b', '[PAD]'], ['c', '[PAD]'], ['[PAD]']]
    tokenized_sequence = tokenizer.convert_tokens_to_ids(sequence)

    removed_seq = remove_special_tokens(tokenized_sequence, tokenizer)

    assert_list_equal(removed_seq, tokenizer.convert_tokens_to_ids([['a', 'b'], ['b'], ['c']]))
