from typing import List, Union, Any, Dict

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.factories.util import can_build_with_subsection, build_with_subsection
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
                  build_context: BuildContext
                  ) -> CanBuildResult:
        config_list = build_context.get_current_config_section().get([])

        for config_dict in config_list:
            # FIXME (AD) this disables access to the full config file (might not matter)
            element_config = Config(config_dict, base_path=build_context.get_current_config_section().base_path)
            element_build_context = BuildContext(element_config, build_context.get_context())

            can_build = can_build_with_subsection(self._object_factory, element_build_context)

            if can_build.type != CanBuildResultType.CAN_BUILD:
                return can_build

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, build_context: BuildContext) -> Union[Any, Dict[str, Any], List[Any]]:
        result = []
        config_list = build_context.get_current_config_section().get([])

        for config_dict in config_list:
            # FIXME (AD) this disables access to the full config file (might not matter)
            element_config = Config(config_dict, base_path=build_context.get_current_config_section().base_path)
            element_build_context = BuildContext(element_config, build_context.get_context())
            single_result = build_with_subsection(self._object_factory, element_build_context)
            result.append(single_result)
        return result

    def is_required(self, build_context: BuildContext) -> bool:
        return self._is_required

    def config_path(self) -> List[str]:
        return self._config_path

    def config_key(self) -> str:
        return self._config_key
