from typing import List

from asme.core.evaluation.evaluation import LogInputEvaluator, ExtractRecommendationEvaluator, ExtractScoresEvaluator
from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.data.datasets.processors.no_target_extractor import NoTargetExtractorProcessor


class ExtractScoresEvaluatorFactory(ObjectFactory):
    """
    Factory for the ExtractScoresEvaluator
    """

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, build_context: BuildContext) -> ExtractScoresEvaluator:
        config = build_context.get_current_config_section()
        context = build_context.get_context()

        item_tokenizer = context.get("tokenizers.item")

        selected_items = context.get("evaluation")["filter_items"].get_selected_items()
        num_predictions = config.get("number_predictions")

        return ExtractScoresEvaluator(item_tokenizer=item_tokenizer, num_predictions=num_predictions,
                                      selected_items=selected_items)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'evaluation'
