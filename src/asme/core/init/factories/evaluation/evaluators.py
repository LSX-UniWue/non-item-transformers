from typing import List

from asme.core.evaluation.evaluation import BatchEvaluator
from asme.core.evaluation.registry import REGISTERED_EVALUATORS
from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.factories.common.conditional_based_factory import ConditionalFactory
from asme.core.init.factories.common.config_based_factory import ListFactory
from asme.core.init.factories.util import can_build_with_subsection, build_with_subsection

from asme.core.init.object_factory import ObjectFactory, CanBuildResult


class EvaluatorsFactory(ObjectFactory):
    """
    Builds all meta information within the `features` section.
    """
    KEY = "evaluators"

    def __init__(self):
        super().__init__()

        self.evaluator_factories = ListFactory(
            ConditionalFactory(
                'type',
                REGISTERED_EVALUATORS
            )
        )


    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        return can_build_with_subsection(self.evaluator_factories, build_context)

    def build(self, build_context: BuildContext) -> List[BatchEvaluator]:
        return build_with_subsection(self.evaluator_factories, build_context)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return ['evaluators']

    def config_key(self) -> str:
        return self.KEY
