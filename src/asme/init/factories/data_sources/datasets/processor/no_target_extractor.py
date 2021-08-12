from typing import List

from asme.init.config import Config
from asme.init.context import Context
from asme.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from data.datasets.processors.no_target_extractor import NoTargetExtractorProcessor


class NoTargetExtractorProcessorFactory(ObjectFactory):
    """
    Factory for the NoTargetExtractorProcessor
    """

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> NoTargetExtractorProcessor:

        return NoTargetExtractorProcessor()

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'no_target_extractor'
