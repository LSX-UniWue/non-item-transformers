from typing import List

from asme.core.init.factories import BuildContext
from asme.core.init.factories.util import check_config_keys_exist
from asme.data.datasets.processors.negative_item_sampler import NegativeItemSamplerProcessor
from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories.features.tokenizer_factory import get_tokenizer_key_for_voc, ITEM_TOKENIZER_ID
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class NegativeItemSamplerProcessorFactory(ObjectFactory):
    """
    factory for the NegativeItemSamplerProcessor
    """

    def can_build(self,
                  build_context: BuildContext
                  ) -> CanBuildResult:

        config = build_context.get_current_config_section()
        context = build_context.get_context()

        config_keys_exist = check_config_keys_exist(config, ['number_negative_items'])
        if not config_keys_exist:
            return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION, 'number_negative_items missing')
        if not context.has_path(get_tokenizer_key_for_voc(ITEM_TOKENIZER_ID)):
            return CanBuildResult(CanBuildResultType.MISSING_DEPENDENCY, 'item tokenizer missing')

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              build_context: BuildContext
              ) -> NegativeItemSamplerProcessor:
        config = build_context.get_current_config_section()
        context = build_context.get_context()

        tokenizer = context.get(get_tokenizer_key_for_voc(ITEM_TOKENIZER_ID))
        num_neg_items = config.get_or_default('number_negative_items', None)

        return NegativeItemSamplerProcessor(tokenizer, num_neg_items)

    def is_required(self, build_context: BuildContext) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'neg_sampler_processor'
