from typing import List, Union, Any, Dict

from init.config import Config
from init.context import Context
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class ConditionalFactory(ObjectFactory):

    """
    a factory that loops through a list and based on a config value call the proper factory
    for the config
    """

    def __init__(self,
                 key: str,
                 factory_dict: Dict[str, ObjectFactory],
                 config_key: str = "",
                 config_path: List[str] = [],
                 is_required: bool = True,
                 ):
        super().__init__()
        self._key = key
        self._factory_dict = factory_dict

        self._is_required = is_required
        self._config_path = config_path
        self._config_key = config_key

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        factory = self._get_factory(config)
        return factory.can_build(config, context)

    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        factory = self._get_factory(config)
        result = factory.build(config, context)
        return result

    def _get_factory(self, config):
        config_value = config.get(self._key)
        if config_value not in self._factory_dict:
            raise ValueError(f'no factory found for {config_value}')
        return self._factory_dict[config_value]

    def is_required(self, context: Context) -> bool:
        return self._is_required

    def config_path(self) -> List[str]:
        return self._config_path

    def config_key(self) -> str:
        return self._config_key
