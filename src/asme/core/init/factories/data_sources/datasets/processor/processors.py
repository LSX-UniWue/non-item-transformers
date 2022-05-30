from typing import List

from asme.core.init.factories import BuildContext
from asme.core.init.factories.util import can_build_with_subsection, build_with_subsection
from asme.data.datasets.processors.processor import Processor
from asme.core.init.context import Context
from asme.core.init.factories.common.conditional_based_factory import ConditionalFactory
from asme.core.init.factories.common.config_based_factory import ListFactory
from asme.core.init.object_factory import ObjectFactory, CanBuildResult
from asme.data.datasets.processors.registry import REGISTERED_PREPROCESSORS

FIXED_SEQUENCE_LENGTH_PROCESSOR_KEY = 'fixed_sequence_length_processor'


class ProcessorsFactory(ObjectFactory):
    KEY = 'processors'

    def __init__(self):
        super().__init__()

        # FIXME: register other processors
        self.processors_factories = ListFactory(
            ConditionalFactory(
                'type',
                REGISTERED_PREPROCESSORS
            )
        )

    def can_build(self,
                  build_context: BuildContext
                  ) -> CanBuildResult:
        return can_build_with_subsection(self.processors_factories, build_context)

    def build(self,
              build_context: BuildContext
              ) -> List[Processor]:
        return build_with_subsection(self.processors_factories, build_context)

    def is_required(self, build_context: BuildContext) -> bool:
        return True

    def config_path(self) -> List[str]:
        return ['processors']

    def config_key(self) -> str:
        return self.KEY
