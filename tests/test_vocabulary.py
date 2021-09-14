import pytest

from asme.core.tokenization.vocabulary import Vocabulary
from collections import OrderedDict

@pytest.fixture
def base_vocabulary():
    sample_vocabulary = OrderedDict([("a", 0), ("b", 1), ("c", 2)])
    vocabulary = Vocabulary(sample_vocabulary)

    return vocabulary


def test_get_id(base_vocabulary):
    assert base_vocabulary.get_id("a") == 0
    assert base_vocabulary.get_id("b") == 1
    assert base_vocabulary.get_id("c") == 2

    assert base_vocabulary.get_id("z") is None


def test_length(base_vocabulary):
    assert len(base_vocabulary) == 3


def test_get_token(base_vocabulary):
    assert base_vocabulary.get_token(0) == "a"
    assert base_vocabulary.get_token(1) == "b"
    assert base_vocabulary.get_token(2) == "c"

    assert base_vocabulary.get_token(10) is None


def test_tokens(base_vocabulary):
    assert base_vocabulary.tokens() == ["a", "b", "c"]


def test_ids(base_vocabulary):
    assert base_vocabulary.ids() == [0, 1, 2]