from typing import List

from asme.data.datasets import ITEM_SEQ_ENTRY_NAME
from asme.data.datasets.processors.tokenizer import TokenizerProcessor
from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories.features.tokenizer_factory import get_tokenizer_key_for_voc, TOKENIZERS_PREFIX,\
    ITEM_TOKENIZER_ID
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class TokenizerProcessorFactory(ObjectFactory):

    """
    Factory for the TokenizerProcessor
    """

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        if not context.has_path(get_tokenizer_key_for_voc(ITEM_TOKENIZER_ID)):
            return CanBuildResult(CanBuildResultType.MISSING_DEPENDENCY, 'item tokenizer missing')

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> TokenizerProcessor:

        tokenizers = context.as_dict()

        tokenizers_map = {}
        for name, tokenizer in tokenizers.items():
            if name.startswith(TOKENIZERS_PREFIX):
                keys_to_tokenize = name.replace(TOKENIZERS_PREFIX + ".", "")
                tokenizers_map[ITEM_SEQ_ENTRY_NAME if keys_to_tokenize == ITEM_TOKENIZER_ID else keys_to_tokenize] = tokenizer

        return TokenizerProcessor(tokenizers_map)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'tokenizer_processor'
