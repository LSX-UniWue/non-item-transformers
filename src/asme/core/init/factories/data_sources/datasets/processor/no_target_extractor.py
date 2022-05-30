from typing import List

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.data.datasets.processors.no_target_extractor import NoTargetExtractorProcessor


class NoTargetExtractorProcessorFactory(ObjectFactory):
    """
    Factory for the NoTargetExtractorProcessor
    """

    def can_build(self,
                  build_context: BuildContext
                  ) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              build_context: BuildContext
              ) -> NoTargetExtractorProcessor:

        return NoTargetExtractorProcessor()

    def is_required(self, build_context: BuildContext) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'no_target_extractor'
