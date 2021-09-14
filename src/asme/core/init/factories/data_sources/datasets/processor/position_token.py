from typing import List

from asme.data.datasets.processors.position_token import PositionTokenProcessor
from asme.core.init.config import Config
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
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:

        config_keys_exist = check_config_keys_exist(config, [self.SEQ_LENGTH_KEY])
        if not config_keys_exist:
            return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION)

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> PositionTokenProcessor:

        seq_length = config.get(self.SEQ_LENGTH_KEY)

        return PositionTokenProcessor(seq_length)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'position_token_processor'
