from typing import Dict, Any

from asme.data.datasets.processors.processor import Processor
from asme.core.tokenization.tokenizer import Tokenizer


class PadFeatureProcessor(Processor):

    """
    Pads a feature to the specified length.

    """

    def __init__(self,
                 tokenizer: Tokenizer,
                 feature_name: str,
                 pad_length: int
                 ):
        """
        :param tokenizers: all tokenizers
        :param feature_name: a feature name
        :param pad_length: the length after padding.
        """
        super().__init__()
        self.pad_length = pad_length
        self.feature_name = feature_name
        self._tokenizer = tokenizer

    def process(self,
                parsed_sequence: Dict[str, Any]
                ) -> Dict[str, Any]:

        pad_token_id = self._tokenizer.pad_token_id
        feature_sequence = parsed_sequence[self.feature_name]
        padding = (self.pad_length - len(feature_sequence)) * [pad_token_id]
        parsed_sequence[self.feature_name] = feature_sequence + padding

        return parsed_sequence
