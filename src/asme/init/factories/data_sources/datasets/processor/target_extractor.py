from typing import List

from asme.init.config import Config
from asme.init.context import Context
from asme.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from data.datasets.processors.target_extractor import TargetExtractorProcessor


class TargetExtractorProcessorFactory(ObjectFactory):
    """
    Factory for the PositionTokenProcessor
    """

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> TargetExtractorProcessor:

        return TargetExtractorProcessor()

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'target_extractor'
