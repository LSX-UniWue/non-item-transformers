from typing import List

from asme.core.evaluation.evaluation import LogInputEvaluator
from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class LogInputEvaluatorFactory(ObjectFactory):
    """
    Factory for the LogInputEvaluator
    """

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, build_context: BuildContext) -> LogInputEvaluator:
        config = build_context.get_current_config_section()
        context = build_context.get_context()

        item_tokenizer = context.get("tokenizers.item")
        return LogInputEvaluator(item_tokenizer=item_tokenizer, )

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'input'
