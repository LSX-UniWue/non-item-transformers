from typing import List, Union, Any, Dict

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.factories.util import can_build_with_subsection, build_with_subsection
from asme.core.init.object_factory import ObjectFactory, CanBuildResult

#TODO rename to something like SelectByKeyFactory
class ConditionalFactory(ObjectFactory):

    """
    a factory that loops through a list and based on a config value call the proper factory
    for the config
    """

    def __init__(self,
                 key: str,
                 factory_dict: Dict[str, ObjectFactory],
                 config_key: str = "",
                 config_path: List[str] = None,
                 is_required: bool = True,
                 ):
        super().__init__()
        if config_path is None:
            config_path = []
        self._key = key
        self._factory_dict = factory_dict

        self._is_required = is_required
        self._config_path = config_path
        self._config_key = config_key

    def can_build(self,
                  build_context: BuildContext
                  ) -> CanBuildResult:
        config = build_context.get_current_config_section()
        factory = self._get_factory(config)
        return can_build_with_subsection(factory, build_context)

    def build(self,
              build_context: BuildContext
              ) -> Union[Any, Dict[str, Any], List[Any]]:
        config = build_context.get_current_config_section()
        factory = self._get_factory(config)
        result = build_with_subsection(factory, build_context)
        return result

    def _get_factory(self,
                     config: Config
                     ) -> ObjectFactory:
        config_value = config.get(self._key)
        if config_value not in self._factory_dict:
            raise ValueError(f'no factory found for {config_value}')
        return self._factory_dict[config_value]

    def is_required(self, build_context: BuildContext) -> bool:
        return self._is_required

    def config_path(self) -> List[str]:
        return self._config_path

    def config_key(self) -> str:
        return self._config_key
