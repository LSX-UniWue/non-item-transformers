from typing import List

from asme.core.init.factories import BuildContext
from asme.data.datasets.processors.pos_neg_sampler import PositiveNegativeSamplerProcessor
from asme.core.init.context import Context
from asme.core.init.factories.features.tokenizer_factory import get_tokenizer_key_for_voc, ITEM_TOKENIZER_ID
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.core.init.factories.features.features_factory import FeaturesFactory
from asme.core.init.factories.util import get_all_tokenizers_from_context


class PositiveNegativeSamplerProcessorFactory(ObjectFactory):

    """
    factory for the PositiveNegativeSamplerProcessor
    """

    def can_build(self,
                  build_context: BuildContext
                  ) -> CanBuildResult:
        context = build_context.get_context()

        if not context.has_path(get_tokenizer_key_for_voc(ITEM_TOKENIZER_ID)):
            return CanBuildResult(CanBuildResultType.MISSING_DEPENDENCY, 'item tokenizer missing')

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              build_context: BuildContext
              ) -> PositiveNegativeSamplerProcessor:
        context = build_context.get_context()
        features = context.get([FeaturesFactory.KEY])
        tokenizers = get_all_tokenizers_from_context(context)

        return PositiveNegativeSamplerProcessor(tokenizers, features)

    def is_required(self, build_context: BuildContext) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'pos_neg_sampler_processor'

