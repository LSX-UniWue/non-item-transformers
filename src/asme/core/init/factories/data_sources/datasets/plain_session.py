from pathlib import Path
from typing import List

from asme.data.base.reader import CsvDatasetIndex, CsvDatasetReader
from asme.data.datasets.processors.processor import Processor
from asme.data.datasets.sequence import PlainSequenceDataset, ItemSessionParser
from asme.core.init.config import Config
from asme.core.init.context import Context
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
                       config: Config,
                       context: Context,
                       session_parser: ItemSessionParser,
                       processors: List[Processor]
                       ) -> PlainSequenceDataset:
        csv_file_path = Path(config.get('csv_file'))
        csv_file_index_path = Path(config.get('csv_file_index'))
        index = CsvDatasetIndex(Path(csv_file_index_path))
        reader = CsvDatasetReader(csv_file_path, index)
        return PlainSequenceDataset(reader, session_parser)

    def _can_build_dataset(self,
                           config: Config,
                           context: Context
                           ) -> CanBuildResult:
        return require_config_keys(config, self.REQUIRED_FIELDS)
