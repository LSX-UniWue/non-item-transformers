import pytest

from collections import OrderedDict

from asme.core.tokenization.tokenizer import Tokenizer
from asme.core.tokenization.vocabulary import Vocabulary


@pytest.fixture
def tokenizer_base():
    sample_vocabulary = OrderedDict([("a", 0), ("b", 1), ("c", 2)])
    vocabulary = Vocabulary(sample_vocabulary)

    return Tokenizer(vocabulary)


@pytest.fixture
def tokenizer_with_unk():
    sample_vocabulary = OrderedDict([("[UNK]", 0), ("a", 1), ("b", 2), ("c", 3)])
    vocabulary = Vocabulary(sample_vocabulary)

    return Tokenizer(vocabulary, unk_token="[UNK]")


def test_convert_tokens_to_ids(tokenizer_base: Tokenizer):

    assert tokenizer_base.convert_tokens_to_ids(None) is None
    assert tokenizer_base.convert_tokens_to_ids("a") == 0
    assert tokenizer_base.convert_tokens_to_ids(["a"]) == [0]
    assert tokenizer_base.convert_tokens_to_ids(["a", "b", "c"]) == [0, 1, 2]
    assert tokenizer_base.convert_tokens_to_ids(["b", "c", "a"]) == [1, 2, 0]

    # no UNK token specified, should be None
    assert tokenizer_base.convert_tokens_to_ids("z") is None


def test_convert_tokens_to_ids_with_unk(tokenizer_with_unk: Tokenizer):

    assert tokenizer_with_unk.convert_tokens_to_ids(None) is None
    assert tokenizer_with_unk.convert_tokens_to_ids("z") == 0
    assert tokenizer_with_unk.convert_tokens_to_ids(["z"]) == [0]
    assert tokenizer_with_unk.convert_tokens_to_ids(["a", "c", "z", "b", "y"]) == [1, 3, 0, 2, 0]
