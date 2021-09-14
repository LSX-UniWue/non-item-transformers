from typing import List

from asme.core.init.config import Config
from asme.core.init.context import Context

from asme.core.init.factories.common.dependencies_factory import DependenciesFactory
from asme.core.init.factories.features.vocabulary_factory import VocabularyFactory
from asme.core.init.factories.util import require_config_keys
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.core.tokenization.tokenizer import Tokenizer


class TokenizerFactory(ObjectFactory):
    """
    Builds a single tokenizer entry inside the tokenizers section.
    """
    # (AD) this is special since we can have multiple tokenizers with individual keys/names.
    KEY = "tokenizer"
    SPECIAL_TOKENS_KEY = "special_tokens"

    def __init__(self,
                 dependencies=DependenciesFactory([VocabularyFactory()])
                 ):
        super().__init__()
        self._dependencies = dependencies

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        dependencies_result = self._dependencies.can_build(config, context)
        if dependencies_result.type != CanBuildResultType.CAN_BUILD:
            return dependencies_result

        return require_config_keys(config, [self.SPECIAL_TOKENS_KEY])

    def build(self,
              config: Config,
              context: Context
              ) -> Tokenizer:
        dependencies = self._dependencies.build(config, context)
        special_tokens = self._get_special_tokens(config)

        return Tokenizer(dependencies["vocabulary"], **special_tokens)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY

    def _get_special_tokens(self, config: Config):
        special_tokens_config = config.get_or_default([self.SPECIAL_TOKENS_KEY], {})
        return special_tokens_config


"""
the special key for the item tokenizer that must at least present in the config
"""
ITEM_TOKENIZER_ID = 'item'

TOKENIZERS_PREFIX = 'tokenizers'


def get_tokenizer_key_for_voc(voc_id: str) -> str:
    return f'{TOKENIZERS_PREFIX}.{voc_id}'
