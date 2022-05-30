from typing import List, Union, Any, Dict

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.factories.util import can_build_with_subsection, build_with_subsection
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class UnionFactory(ObjectFactory):
    """
    Selects the appropriate factory to handle a configuration section from a list of possible candidates. If multiple
    factories are eligible to handle the configuration section, the first one as supplied during instantiation is used.
    """

    def __init__(self, factories: List[ObjectFactory], config_key: str, config_path: List[str], required: bool = True):
        super(UnionFactory, self).__init__()
        self._config_key = config_key
        self._config_path = config_path
        self.required = required
        self.factories = factories

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        for result in map(lambda factory: can_build_with_subsection(factory, build_context), self.factories):
            if result.type == CanBuildResultType.CAN_BUILD:
                return result

        return CanBuildResult(
            CanBuildResultType.INVALID_CONFIGURATION,
            f"None of the configured factories can handle the configuration section {'.'.join(build_context.get_current_config_section().base_path)}."
        )

    def build(self, build_context: BuildContext) -> Union[Any, Dict[str, Any], List[Any]]:
        for factory in self.factories:
            can_build_result = factory.can_build(build_context)
            if can_build_result.type == CanBuildResultType.CAN_BUILD:
                return factory.build(build_context)

        raise Exception(f"No factory was able to build the configuration section {build_context.get_current_config_section().get_config([]).config}.")

    def is_required(self, build_context: BuildContext) -> bool:
        return self.required

    def config_path(self) -> List[str]:
        return self._config_path

    def config_key(self) -> str:
        return self._config_key
