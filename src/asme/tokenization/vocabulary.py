import collections
import csv

from typing import List, Optional, OrderedDict, TextIO, Union


class Vocabulary(object):
    def __init__(self,
                 token_to_id: OrderedDict[str, int]
                 ):
        #FIXME (AD): make sure that tokens are strings, otherwise get_id fails to retrieve the correct id
        self.token_to_id = token_to_id
        self.id_to_token = collections.OrderedDict([(id, token) for token, id in token_to_id.items()])

    def get_id(self, token: Union[int, str]) -> Optional[int]:
        if isinstance(token, int):
            token = str(token)

        if token not in self.token_to_id:
            return None

        return self.token_to_id[token]

    def get_token(self, id: int) -> Optional[str]:
        if id not in self.id_to_token:
            return None

        return self.id_to_token[id]

    def tokens(self) -> List[str]:
        return list(map(lambda x: x[0], self.token_to_id.items()))

    def ids(self) -> List[int]:
        return list(map(lambda x: x[1], self.token_to_id.items()))

    def __len__(self):
        return len(self.token_to_id)

    def __iter__(self):
        return iter(self.token_to_id.items())


class VocabularyBuilder(object):
    def __init__(self,
                 tokens: List[str] = None,
                 start_id: int = 0
                 ):
        if tokens is None:
            tokens = []
        self.next_id = start_id + len(tokens)
        self.token_to_id = collections.OrderedDict(zip(tokens, range(start_id, self.next_id)))

    def add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_id
            self.next_id += 1

        return self.token_to_id[token]

    def build(self) -> Vocabulary:
        return Vocabulary(self.token_to_id)


class VocabularyReaderWriter(object):
    def write(self, vocabulary: Vocabulary, output: TextIO):
        raise NotImplementedError()

    def read(self, file_input: TextIO) -> Vocabulary:
        raise NotImplementedError()


class CSVVocabularyReaderWriter(VocabularyReaderWriter):
    """
        Encodes the vocabulary as a CSV data file with every entry following the pattern: <token><delimiter><id>.
    """
    def __init__(self, delimiter: str = "\t"):
        self.delimiter = delimiter

    def write(self, vocabulary: Vocabulary, output: TextIO):
        writer = csv.writer(output, delimiter=self.delimiter)
        for token in vocabulary.tokens():
            writer.writerow([token, vocabulary.get_id(token)])

    def read(self,
             file_input: TextIO
             ) -> Vocabulary:
        reader = csv.reader(file_input, delimiter=self.delimiter)
        vocabulary_entries = [(token, int(id)) for [token, id] in reader]

        return Vocabulary(collections.OrderedDict(vocabulary_entries))


class SequentialIdVocabularyReaderWriter(VocabularyReaderWriter):
    """
        Assumes that the vocabulary consists of consecutively numbered tokens starting with 0. Only writes the tokens
        in this order and recovers the ids on reading by assuming that the id is identical to the line number of the
        token.
    """
    def write(self, vocabulary: Vocabulary, output: TextIO):
        for token in vocabulary.tokens():
            output.write(token)
            output.write("\n")

    def read(self, file_input: TextIO) -> Vocabulary:
        tokens = [token.strip() for token in file_input]
        token_to_id = zip(tokens, range(len(tokens)))

        return Vocabulary(collections.OrderedDict(token_to_id))
