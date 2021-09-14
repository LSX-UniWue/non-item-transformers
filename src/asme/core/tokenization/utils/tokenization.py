from typing import List, Union

from asme.core.tokenization.tokenizer import Tokenizer


def remove_special_tokens(sequence: Union[List[int], List[List[int]]],
                          tokenizer: Tokenizer
                          ) -> Union[List[int], List[List[int]]]:
    is_basket_recommendation = isinstance(sequence[0], list)

    if is_basket_recommendation:
        result = []
        for basket in sequence:
            basket = _remove_special_tokens_from_sequence(basket, tokenizer)
            if len(basket) > 0:
                result.append(basket)

        return result
    return _remove_special_tokens_from_sequence(sequence, tokenizer)


def _remove_special_tokens_from_sequence(sequence: List[int],
                                         tokenizer: Tokenizer
                                         ) -> List[int]:
    return list(filter(lambda token_id: token_id not in tokenizer.get_special_token_ids(), sequence))
