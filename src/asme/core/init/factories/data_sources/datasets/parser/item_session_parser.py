from pathlib import Path
from typing import Any, List

from asme.core.init.factories import BuildContext
from asme.data.datasets.sequence import ItemSessionParser
from asme.data.utils.csv import create_indexed_header, read_csv_header
from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class ItemSessionParserFactory(ObjectFactory):

    KEY = 'session_parser'

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        # TODO: config validation?
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, build_context: BuildContext) -> Any:
        config = build_context.get_current_config_section()
        context = build_context.get_context()

        csv_file = Path(config.get('csv_file'))
        parser_config = config.get_config(['parser'])
        delimiter = parser_config.get_or_default('delimiter', '\t')

        features = context.get('features')

        header = create_indexed_header(read_csv_header(csv_file, delimiter=delimiter))
        return ItemSessionParser(header, features, delimiter=delimiter)

    def is_required(self, build_context: BuildContext) -> bool:
        return True

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return self.KEY
