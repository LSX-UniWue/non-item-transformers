from typing import Any
from typing import List

from asme.core.init.factories import BuildContext
from asme.core.init.factories.common.dependencies_factory import DependenciesFactory
from asme.core.init.factories.data_sources.datasets.dataset_factory import DatasetFactory
from asme.core.init.factories.data_sources.datasets.processor.processors import ProcessorsFactory
from asme.core.init.factories.util import require_config_field_equal
from asme.core.init.object_factory import CanBuildResult
from asme.core.init.object_factory import CanBuildResultType
from asme.data.datasets.placeholder_dataset import PlaceholderDataset
from asme.data.datasets.processors.processor import Processor
#from asme.core.init.factories.data_sources.datasets.registry import register_dataset_factory

class PlaceholderDatasetFactory(DatasetFactory):

    def __init__(self):
        super(PlaceholderDatasetFactory, self).__init__()
        # Only needs processors factory as dependency, overwrite dependencies from super
        self.parser_dependency = DependenciesFactory([ProcessorsFactory()])

    def _can_build_dataset(self, build_context: BuildContext) -> CanBuildResult:
        result = require_config_field_equal(build_context.get_current_config_section(), 'type', 'placeholder')
        if result.type != CanBuildResultType.CAN_BUILD:
            return result
        else:
            return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def _build_dataset(self,
                       build_context: BuildContext,
                       session_parser: None,
                       processors: List[Processor]) -> Any:

        return PlaceholderDataset(processors)
