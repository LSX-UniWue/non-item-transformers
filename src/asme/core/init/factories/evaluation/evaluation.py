from typing import List

from asme.core.evaluation.evaluation import BatchEvaluator
from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.factories.common.dependencies_factory import DependenciesFactory
from asme.core.init.factories.evaluation.evaluators import EvaluatorsFactory
from asme.core.init.factories.evaluation.item_filter import FilterPredictionItemsFactory
from asme.core.init.factories.evaluation.writers import WritersFactory
from asme.core.init.factories.util import can_build_with_subsection, build_with_subsection

from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class EvaluationFactory(ObjectFactory):
    """
    Builds all meta information within the `evaluation` section.
    """
    KEY = "evaluation"

    def __init__(self):
        super().__init__()

        self.filter_factory = DependenciesFactory([FilterPredictionItemsFactory()])
        self.evaluators_factory = DependenciesFactory([EvaluatorsFactory()])
        self.writers_factory = DependenciesFactory([WritersFactory()])

    def can_build(self, build_context: BuildContext) -> CanBuildResult:

        can_build_result = can_build_with_subsection(self.filter_factory, build_context)
        if can_build_result.type != CanBuildResultType.CAN_BUILD:
            return can_build_result

        can_build_result = can_build_with_subsection(self.evaluators_factory, build_context)
        if can_build_result.type != CanBuildResultType.CAN_BUILD:
            return can_build_result

        can_build_result = can_build_with_subsection(self.writers_factory, build_context)
        if can_build_result.type != CanBuildResultType.CAN_BUILD:
            return can_build_result

        return can_build_result

    def build(self, build_context: BuildContext) -> List[BatchEvaluator]:
        config = build_context.get_current_config_section()
        context = build_context.get_context()

        num_predictions = config.get_or_default("number_predictions", 20)

        for evaluator in config.get("evaluators"):
           evaluator["number_predictions"] = num_predictions

        filter_items = build_with_subsection(self.filter_factory, build_context)
        context.set(self.config_path(), filter_items)

        evaluation = build_with_subsection(self.evaluators_factory, build_context)
        evaluation.update(filter_items)
        context.set(self.config_path(), evaluation, overwrite=True)

        writer = build_with_subsection(self.writers_factory, build_context)
        evaluation.update(writer)
        context.set(self.config_path(), evaluation, overwrite=True)

        return evaluation

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
