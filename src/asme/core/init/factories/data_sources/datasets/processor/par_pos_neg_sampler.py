from typing import List

from asme.data.datasets.processors.par_pos_neg_sampler import ParameterizedPositiveNegativeSamplerProcessor
from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories.features.tokenizer_factory import get_tokenizer_key_for_voc, ITEM_TOKENIZER_ID
from asme.core.init.factories.util import check_config_keys_exist
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class ParameterizedPositiveNegativeSamplerProcessorFactory(ObjectFactory):

    """
    factory for the PositiveNegativeSamplerProcessor
    """

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        config_keys_exist = check_config_keys_exist(config, ['t'])  # FIXME: rename parameter
        if not config_keys_exist:
            return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION)

        if not context.has_path(get_tokenizer_key_for_voc(ITEM_TOKENIZER_ID)):
            return CanBuildResult(CanBuildResultType.MISSING_DEPENDENCY, 'item tokenizer missing')

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> ParameterizedPositiveNegativeSamplerProcessor:
        tokenizer = context.get(get_tokenizer_key_for_voc(ITEM_TOKENIZER_ID))
        t = config.get('t')

        return ParameterizedPositiveNegativeSamplerProcessor(tokenizer, t)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'par_pos_neg_sampler_processor'
