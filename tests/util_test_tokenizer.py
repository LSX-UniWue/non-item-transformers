from pathlib import Path

from asme.core.tokenization.tokenizer import Tokenizer
from asme.core.tokenization.vocabulary import CSVVocabularyReaderWriter

# TODO: move somewhere else
TEST_DATASET_BASE_PATH = Path('tests/example_dataset/')


def create_tokenizer() -> Tokenizer:
    with open(TEST_DATASET_BASE_PATH / 'example.vocabulary.item_id.txt') as vocab_file:
        vocab_reader = CSVVocabularyReaderWriter()
        vocab = vocab_reader.read(vocab_file)
        return Tokenizer(vocab, pad_token='<PAD>', mask_token='<MASK>', unk_token='<UNK>')
