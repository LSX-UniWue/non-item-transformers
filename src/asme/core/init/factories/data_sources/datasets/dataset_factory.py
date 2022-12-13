from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Union

from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.factories.common.dependencies_factory import DependenciesFactory
from asme.core.init.factories.data_sources.datasets.parser.item_session_parser import ItemSessionParserFactory
from asme.core.init.factories.data_sources.datasets.processor.processors import ProcessorsFactory
from asme.core.init.factories.util import build_with_subsection
from asme.core.init.factories.util import can_build_with_subsection
from asme.core.init.object_factory import CanBuildResult
from asme.core.init.object_factory import CanBuildResultType
from asme.core.init.object_factory import ObjectFactory
from asme.data.datasets.processors.processor import Processor
from asme.data.datasets.sequence import ItemSessionParser


class DatasetFactory(ObjectFactory):

    KEY = "dataset"

    def __init__(self):
        super().__init__()
        self.parser_dependency = DependenciesFactory([ItemSessionParserFactory(), ProcessorsFactory()])

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        can_build = can_build_with_subsection(self.parser_dependency, build_context)
        if can_build.type != CanBuildResultType.CAN_BUILD:
            return can_build

        return self._can_build_dataset(build_context)

    def build(self, build_context: BuildContext) -> Union[Any, Dict[str, Any], List[Any]]:
        build_result = build_with_subsection(self.parser_dependency, build_context)

        session_parser = build_result.get(ItemSessionParserFactory.KEY, None)
        processors = build_result[ProcessorsFactory.KEY]

        return self._build_dataset(build_context, session_parser, processors)

    def is_required(self, build_context: BuildContext) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY

    @abstractmethod
    def _build_dataset(self,
                       build_context: BuildContext,
                       session_parser: ItemSessionParser,
                       processors: List[Processor]
                       ) -> Any:
        pass

    @abstractmethod
    def _can_build_dataset(self,
                           build_context: BuildContext
                           ) -> CanBuildResult:
        pass
