from typing import List

from asme.core.evaluation.evaluation import PerSampleMetricsEvaluator
from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class PerSampleMetricsEvaluatorFactory(ObjectFactory):
    """
    Factory for the ExtractRecommendationEvaluator
    """

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, build_context: BuildContext) -> PerSampleMetricsEvaluator:
        config = build_context.get_current_config_section()
        context = build_context.get_context()
        item_tokenizer = context.get("tokenizers.item")
        selected_items = context.get("evaluation")["filter_items"].get_selected_items()
        module = context.get("module")

        return PerSampleMetricsEvaluator(item_tokenizer=item_tokenizer, selected_items=selected_items,
                                         module=module)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'evaluation'
