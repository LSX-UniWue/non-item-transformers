from typing import List, Union, Any, Dict

from asme.data.datasets.processors.processor import Processor
from asme.data.datasets.sequence import ItemSessionParser
from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories.common.dependencies_factory import DependenciesFactory
from asme.core.init.factories.data_sources.datasets.parser.item_session_parser import ItemSessionParserFactory
from asme.core.init.factories.data_sources.datasets.processor.processors import ProcessorsFactory
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType

from abc import abstractmethod


class DatasetFactory(ObjectFactory):

    KEY = "dataset"

    def __init__(self):
        super().__init__()
        self.parser_dependency = DependenciesFactory([ItemSessionParserFactory(), ProcessorsFactory()])

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        can_build = self.parser_dependency.can_build(config, context)
        if can_build.type != CanBuildResultType.CAN_BUILD:
            return can_build

        return self._can_build_dataset(config, context)

    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        build_result = self.parser_dependency.build(config, context)

        session_parser = build_result[ItemSessionParserFactory.KEY]
        processors = build_result[ProcessorsFactory.KEY]

        return self._build_dataset(config, context, session_parser, processors)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY

    @abstractmethod
    def _build_dataset(self,
                       config: Config,
                       context: Context,
                       session_parser: ItemSessionParser,
                       processors: List[Processor]
                       ) -> Any:
        pass

    @abstractmethod
    def _can_build_dataset(self,
                           config: Config,
                           context: Context
                           ) -> CanBuildResult:
        pass


