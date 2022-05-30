from pathlib import Path
from typing import List

from asme.core.init.factories import BuildContext
from asme.core.init.factories.data_sources.datasets.processor.last_item_mask import get_sequence_feature_names
from asme.data.datasets.sequence_position import SequencePositionIndex
from asme.data.datasets.sequence_position import SequencePositionDataset
from asme.data.datasets.processors.processor import Processor
from asme.data.datasets.sequence import ItemSessionParser
from asme.core.init.factories.data_sources.datasets.plain_session import PlainSessionDatasetFactory
from asme.core.init.factories.util import require_config_field_equal, require_config_keys, can_build_with_subsection, \
    build_with_subsection
from asme.core.init.object_factory import CanBuildResult, CanBuildResultType
from asme.core.init.factories.data_sources.datasets.dataset_factory import DatasetFactory


class SequencePositionDatasetFactory(DatasetFactory):

    REQUIRED_FIELDS = ["nip_index_file"]

    def __init__(self):
        super(SequencePositionDatasetFactory, self).__init__()
        self.plain_session_factory = PlainSessionDatasetFactory()

    def _can_build_dataset(self, build_context: BuildContext) -> CanBuildResult:
        result = require_config_field_equal(build_context.get_current_config_section(), 'type', 'sequence_position')
        if result.type != CanBuildResultType.CAN_BUILD:
            return result

        result = self.plain_session_factory.can_build(build_context)
        if result.type != CanBuildResultType.CAN_BUILD:
            return result

        return require_config_keys(build_context.get_current_config_section(), self.REQUIRED_FIELDS)

    def _build_dataset(self,
                       build_context: BuildContext,
                       session_parser: ItemSessionParser,
                       processors: List[Processor]
                       ) -> SequencePositionDataset:
        nip_index_file_path = Path(build_context.get_current_config_section().get('nip_index_file'))
        basic_dataset = self.plain_session_factory.build(build_context)

        sequences_to_truncate = get_sequence_feature_names(build_context.get_current_config_section(), build_context.get_context())

        return SequencePositionDataset(basic_dataset,
                                       SequencePositionIndex(nip_index_file_path),
                                       sequences_to_truncate=sequences_to_truncate,
                                       processors=processors)
