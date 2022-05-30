from pathlib import Path
from typing import List

from asme.core.init.factories import BuildContext
from asme.data.base.reader import CsvDatasetIndex, CsvDatasetReader
from asme.data.datasets.processors.processor import Processor
from asme.data.datasets.sequence import PlainSequenceDataset, ItemSessionParser
from asme.core.init.factories.data_sources.datasets.dataset_factory import DatasetFactory
from asme.core.init.factories.util import require_config_keys
from asme.core.init.object_factory import CanBuildResult


class PlainSessionDatasetFactory(DatasetFactory):

    """
    Factory to build a plain sequence dataset
    """

    REQUIRED_FIELDS = ["csv_file", "csv_file_index"]

    def __init__(self):
        super(PlainSessionDatasetFactory, self).__init__()

    def _build_dataset(self,
                       build_context: BuildContext,
                       session_parser: ItemSessionParser,
                       processors: List[Processor]
                       ) -> PlainSequenceDataset:
        config = build_context.get_current_config_section()
        csv_file_path = Path(config.get('csv_file'))
        csv_file_index_path = Path(config.get('csv_file_index'))
        index = CsvDatasetIndex(Path(csv_file_index_path))
        reader = CsvDatasetReader(csv_file_path, index)
        return PlainSequenceDataset(reader, session_parser)

    def _can_build_dataset(self,
                           build_context: BuildContext
                           ) -> CanBuildResult:
        return require_config_keys(build_context.get_current_config_section(), self.REQUIRED_FIELDS)
