import torch

from typing import Union, List, Dict, Any, Set

from asme.data.datasets import ITEM_SEQ_ENTRY_NAME, POSITIVE_SAMPLES_ENTRY_NAME, NEGATIVE_SAMPLES_ENTRY_NAME
from asme.data.datasets.processors.processor import Processor
from asme.core.tokenization.tokenizer import Tokenizer


class NegativeItemSamplerProcessor(Processor):

    """
    Takes the input sequence and samples {num_neg_items} negative items
    """

    def __init__(self,
                 tokenizer: Tokenizer,
                 num_neg_items: int = 1
                 ):
        super().__init__()
        self._tokenizer = tokenizer
        self.num_neg_items = num_neg_items

    def _sample_negative_items(self,
                               session: Union[List[int], List[List[int]]]
                               ) -> Union[List[int], List[List[int]]]:

        # we want to use a multinomial distribution to draw negative samples fast
        # start with a uniform distribution over all vocabulary tokens
        weights = torch.ones([len(self._tokenizer)])

        # set weight for special tokens to 0.
        weights[self._tokenizer.get_special_token_ids()] = 0.

        # prevent sampling of tokens already present in the session
        used_tokens = self._get_all_tokens_of_session(session)
        weights[list(used_tokens)] = 0.

        if isinstance(session[0], list):
            results = []
            for seq_step in session[:self.num_neg_items]:  # skip last num_neg_items sequence steps
                neg_samples = torch.multinomial(weights, num_samples=self.num_neg_items, replacement=True).tolist()
                results.append(neg_samples)
            return results

        return torch.multinomial(weights, num_samples=self.num_neg_items, replacement=True).tolist()

    def _get_all_tokens_of_session(self, session: Union[List[int], List[List[int]]],
                                   ) -> Set[int]:
        if isinstance(session[0], list):
            flat_items = [item for sublist in session for item in sublist]
        else:
            flat_items = session

        return set(flat_items)

    def process(self, parsed_session: Dict[str, Any]) -> Dict[str, Any]:
        session = parsed_session[ITEM_SEQ_ENTRY_NAME]

        if POSITIVE_SAMPLES_ENTRY_NAME not in parsed_session:
            raise ValueError("positive items missing")

        neg = self._sample_negative_items(session)

        parsed_session[NEGATIVE_SAMPLES_ENTRY_NAME] = neg
        return parsed_session
