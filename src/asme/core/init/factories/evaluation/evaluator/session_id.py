from typing import List

from asme.core.evaluation.evaluation import ExtractSampleIdEvaluator
from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType

class ExtractSampleIdEvaluatorFactory(ObjectFactory):
    """
    Factory for the ExtractSampleIdEvaluator
    """

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, build_context: BuildContext) -> ExtractSampleIdEvaluator:
        config = build_context.get_current_config_section()

        use_session_id = config.get_or_default("use_session_id", False)
        return ExtractSampleIdEvaluator(use_session_id=use_session_id)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'recommendation'
