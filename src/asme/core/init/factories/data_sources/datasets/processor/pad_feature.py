from typing import List

from asme.core.init.factories import BuildContext
from asme.core.init.factories.features.tokenizer_factory import get_tokenizer_key_for_voc
from asme.core.init.factories.util import check_config_keys_exist
from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.data.datasets.processors.pad_feature import PadFeatureProcessor


class PadFeatureProcessorFactory(ObjectFactory):

    """
    Factory for the PadFeatureProcessor
    """

    def can_build(self,
                  build_context: BuildContext
                  ) -> CanBuildResult:

        config = build_context.get_current_config_section()
        context = build_context.get_context()
        config_keys_exist = check_config_keys_exist(config, ['feature_name', 'pad_length'])

        if not config_keys_exist:
            return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION)

        if not context.has_path(get_tokenizer_key_for_voc(config.get('feature_name'))):
            return CanBuildResult(CanBuildResultType.MISSING_DEPENDENCY, f"{config.get('feature_name')} tokenizer missing")

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              build_context: BuildContext
              ) -> PadFeatureProcessor:
        config = build_context.get_current_config_section()
        context = build_context.get_context()

        feature_name = config.get('feature_name')
        pad_length = config.get('pad_length')
        tokenizer = context.get(get_tokenizer_key_for_voc(feature_name))
        return PadFeatureProcessor(tokenizer, feature_name, pad_length)

    def is_required(self, build_context: BuildContext) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'pad_feature_processor'
