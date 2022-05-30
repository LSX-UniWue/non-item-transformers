from typing import List

from asme.core.init.factories import BuildContext
from asme.data.datasets.processors.position_token import PositionTokenProcessor
from asme.core.init.context import Context
from asme.core.init.factories.util import check_config_keys_exist
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class PositionTokenProcessorFactory(ObjectFactory):
    #FIXME make this configurable for other tokenizers, e.g. keywords
    SEQ_LENGTH_KEY = 'seq_length'
    """
    Factory for the PositionTokenProcessor
    """

    def can_build(self,
                  build_context: BuildContext
                  ) -> CanBuildResult:

        config_keys_exist = check_config_keys_exist(build_context.get_current_config_section(), [self.SEQ_LENGTH_KEY])
        if not config_keys_exist:
            return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION)

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              build_context: BuildContext
              ) -> PositionTokenProcessor:

        seq_length = build_context.get_current_config_section().get(self.SEQ_LENGTH_KEY)

        return PositionTokenProcessor(seq_length)

    def is_required(self, build_context: BuildContext) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'position_token_processor'
