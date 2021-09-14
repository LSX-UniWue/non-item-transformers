from pathlib import Path
from typing import List

from asme.core.init.factories.data_sources.datasets.processor.last_item_mask import get_sequence_feature_names
from asme.data.datasets.sequence_position import SequencePositionIndex
from asme.data.datasets.sequence_position import SequencePositionDataset
from asme.data.datasets.processors.processor import Processor
from asme.data.datasets.sequence import ItemSessionParser
from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories.data_sources.datasets.plain_session import PlainSessionDatasetFactory
from asme.core.init.factories.util import require_config_field_equal, require_config_keys
from asme.core.init.object_factory import CanBuildResult, CanBuildResultType
from asme.core.init.factories.data_sources.datasets.dataset_factory import DatasetFactory


class SequencePositionDatasetFactory(DatasetFactory):

    REQUIRED_FIELDS = ["nip_index_file"]

    def __init__(self):
        super(SequencePositionDatasetFactory, self).__init__()
        self.plain_session_factory = PlainSessionDatasetFactory()

    def _can_build_dataset(self, config: Config, context: Context) -> CanBuildResult:
        result = require_config_field_equal(config, 'type', 'sequence_position')
        if result.type != CanBuildResultType.CAN_BUILD:
            return result

        result = self.plain_session_factory.can_build(config, context)
        if result.type != CanBuildResultType.CAN_BUILD:
            return result

        return require_config_keys(config, self.REQUIRED_FIELDS)

    def _build_dataset(self,
                       config: Config,
                       context: Context,
                       session_parser: ItemSessionParser,
                       processors: List[Processor]
                       ) -> SequencePositionDataset:
        nip_index_file_path = Path(config.get('nip_index_file'))
        basic_dataset = self.plain_session_factory.build(config, context)

        sequences_to_truncate = get_sequence_feature_names(config, context)

        return SequencePositionDataset(basic_dataset,
                                       SequencePositionIndex(nip_index_file_path),
                                       sequences_to_truncate=sequences_to_truncate,
                                       processors=processors)
