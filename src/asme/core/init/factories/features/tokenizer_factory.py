from typing import List

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories import BuildContext

from asme.core.init.factories.common.dependencies_factory import DependenciesFactory
from asme.core.init.factories.features.vocabulary_factory import VocabularyFactory
from asme.core.init.factories.util import require_config_keys, can_build_with_subsection, build_with_subsection
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

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        dependencies_result = can_build_with_subsection(self._dependencies, build_context)
        if dependencies_result.type != CanBuildResultType.CAN_BUILD:
            return dependencies_result

        return require_config_keys(build_context.get_current_config_section(), [self.SPECIAL_TOKENS_KEY])

    def build(self, build_context: BuildContext) -> Tokenizer:
        dependencies = build_with_subsection(self._dependencies, build_context)
        special_tokens = self._get_special_tokens(build_context.get_current_config_section())

        return Tokenizer(dependencies["vocabulary"], **special_tokens)

    def is_required(self, build_context: BuildContext) -> bool:
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
