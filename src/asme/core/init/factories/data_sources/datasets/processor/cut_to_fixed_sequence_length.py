from typing import List

from asme.core.init.factories.util import check_config_keys_exist
from asme.data.datasets.processors.cut_to_fixed_sequence_length import CutToFixedSequenceLengthProcessor
from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class CutToFixedSequenceLengthProcessorFactory(ObjectFactory):

    """
    factory for the PositiveNegativeSamplerProcessor
    """

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        config_keys_exist = check_config_keys_exist(config, ['fixed_length'])
        if not config_keys_exist:
            return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION)
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> CutToFixedSequenceLengthProcessor:
        fixed_length = config.get('fixed_length')
        return CutToFixedSequenceLengthProcessor(fixed_length)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'fixed_sequence_length_processor'
