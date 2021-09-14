from typing import Dict, Any

from asme.data.datasets.processors.processor import Processor
from asme.core.tokenization.tokenizer import Tokenizer


class TokenizerProcessor(Processor):

    """
    Tokenizes the the fields specified by the tokenizer id of the sequence

    Must be the first tokenizer in the chain of processors

    Example:
        Input:
            sequence: [item 1, item 2, item 5, item 22]
        Output:
            sequence: [1, 5, 7, 8]

        where 1, 5, 7 and 8 are the token ids of the corresponding items in the sequence

    """

    def __init__(self,
                 tokenizers: Dict[str, Tokenizer]
                 ):
        """
        :param tokenizers: the tokenizer to use for the tokenization
        """
        super().__init__()
        self._tokenizers = tokenizers

    def process(self,
                parsed_sequence: Dict[str, Any]
                ) -> Dict[str, Any]:
        for tokenizer_key, tokenizer in self._tokenizers.items():
            if tokenizer_key in parsed_sequence:
                to_tokenize = parsed_sequence[tokenizer_key]
                tokenized = tokenizer.convert_tokens_to_ids(to_tokenize)
                parsed_sequence[tokenizer_key] = tokenized

        return parsed_sequence
