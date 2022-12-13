from typing import Any
from typing import List

from asme.core.init.factories import BuildContext
from asme.core.init.factories.data_sources.datasets.dataset_factory import DatasetFactory
from asme.core.init.factories.data_sources.datasets.plain_session import PlainSessionDatasetFactory
from asme.core.init.factories.util import require_config_field_equal
from asme.core.init.object_factory import CanBuildResult
from asme.core.init.object_factory import CanBuildResultType
from asme.data.datasets.processors.processor import Processor
from asme.data.datasets.sequence import ItemSequenceDataset
from asme.data.datasets.sequence import ItemSessionParser


class ItemSessionDatasetFactory(DatasetFactory):

    def __init__(self):
        super(ItemSessionDatasetFactory, self).__init__()
        self.plain_session_factory = PlainSessionDatasetFactory()

    def _can_build_dataset(self, build_context: BuildContext) -> CanBuildResult:
        result = require_config_field_equal(build_context.get_current_config_section(), 'type', 'session')
        if result.type != CanBuildResultType.CAN_BUILD:
            return result

        return self.plain_session_factory.can_build(build_context)

    def _build_dataset(self,
                       build_context: BuildContext,
                       session_parser: ItemSessionParser,
                       processors: List[Processor]) -> Any:
        basic_dataset = self.plain_session_factory.build(build_context)

        return ItemSequenceDataset(basic_dataset, processors=processors)
