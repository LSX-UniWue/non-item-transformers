from typing import List, Union, Any, Dict

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class ListFactory(ObjectFactory):

    """
    a factory that loops through a list and calls the object factory on this sub config
    """

    def __init__(self,
                 object_factory: ObjectFactory,
                 config_key: str = "",
                 config_path: List[str] = None,
                 is_required: bool = True,
                 ):
        super().__init__()
        if config_path is None:
            config_path = []
        self._object_factory = object_factory

        self._is_required = is_required
        self._config_path = config_path
        self._config_key = config_key

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        config_list = config.get([])

        for config_dict in config_list:
            config = Config(config_dict, base_path=config.base_path)
            can_build = self._object_factory.can_build(config, context)
            if can_build.type != CanBuildResultType.CAN_BUILD:
                return can_build

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        result = []
        config_list = config.get([])

        for config_dict in config_list:
            config = Config(config_dict, base_path=config.base_path)
            single_result = self._object_factory.build(config, context)
            result.append(single_result)
        return result

    def is_required(self, context: Context) -> bool:
        return self._is_required

    def config_path(self) -> List[str]:
        return self._config_path

    def config_key(self) -> str:
        return self._config_key
