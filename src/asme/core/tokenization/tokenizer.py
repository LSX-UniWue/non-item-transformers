from asme.core.tokenization import SPECIAL_TOKENS_ATTRIBUTES
from asme.core.tokenization.vocabulary import Vocabulary

from typing import List, Optional, Union


class Tokenizer:
    """

    TODO: add docu

    """

    def __init__(self,
                 vocabulary: Vocabulary,
                 **kwargs
                 ):
        self.vocabulary = vocabulary

        self.bos_token = None
        self.eos_token = None
        self.unk_token = None
        self.mask_token = None
        self.pad_token = None
        self.sep_token = None
        self.cls_token = None

        for key, value in kwargs.items():
            if key in SPECIAL_TOKENS_ATTRIBUTES:
                assert isinstance(value, str)
                setattr(self, key, value)

    @property
    def pad_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def unk_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def mask_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def bos_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def sep_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def cls_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.cls_token)

    def get_special_token_ids(self) -> List[int]:
        """
        returns a list of all special token ids
        :return: list of all special token ids
        """
        special_token_ids = []
        for key in SPECIAL_TOKENS_ATTRIBUTES:
            token = getattr(self, key)
            if token is not None:
                token_id = self.convert_tokens_to_ids(token)
                special_token_ids.append(token_id)

        return special_token_ids

    def get_vocabulary(self) -> Vocabulary:
        """
        :return: the vocabulary of the tokenizer
        """
        return self.vocabulary

    def convert_ids_to_tokens(self,
                              token_ids: Union[int, List[int], List[List[int]]]
                              ) -> Union[Optional[str], List[str], List[List[str]]]:
        if token_ids is None:
            return None

        if isinstance(token_ids, int):
            return self._convert_id_to_item(token_ids)

        items = []
        for token_id in token_ids:
            items.append(self._convert_id_to_item(token_id))
        return items

    def convert_tokens_to_ids(self,
                              items: Union[str, List[str], List[List[str]]]
                              ) -> Union[Optional[int], List[int]]:
        if items is None:
            return None

        if isinstance(items, str):
            return self._convert_item_to_id(items)

        ids = []
        for item in items:
            ids.append(self._convert_item_to_id(item))
        return ids

    def _convert_id_to_item(self,
                            token_id: Union[int, List[int]]
                            ) -> Union[Optional[str], List[str]]:
        """
        Converts the token_id into its token.

        :param token_id: a token_id
        :return: the token if the token is part of the vocabulary
        """
        if token_id is None:
            return None

        if isinstance(token_id, int):
            token = self.vocabulary.get_token(token_id)
            if token is None:
                return None
            return token

        # here we assume it is a list
        tokens = []
        for t_id in token_id:
            token = self._convert_id_to_item(t_id)
            tokens.append(token)
        return tokens

    # FIXME (AD) if the vocabulary does not contain an UNK token, this will cause an endless recursion.
    def _convert_item_to_id(self,
                            token: Union[str, List[str]]
                            ) -> Union[Optional[int], List[int]]:
        """
        Converts the token into its id. If the token is not part of the vocabulary and the unk_token property is set,
        the id for the unk_token will be returned. Otherwise if the token can not be found, None is returned.

        :param token: a token
        :return: the token id if the token is part of the vocabulary, otherwise if possible the id of the unk_token is
         returned else the return value is None
        """
        if token is None:
            return None

        if isinstance(token, str):
            token_id = self.vocabulary.get_id(token)
            if token_id is None:
                if self.unk_token is not None:
                    return self.unk_token_id
            return token_id

        # here we assume it is a list
        ids = []
        for item in token:
            encoded_token = self._convert_item_to_id(item)
            ids.append(encoded_token)
        return ids

    def __len__(self):
        """ Size of the full vocabulary with the added tokens """
        return len(self.vocabulary)
