from typing import List, Union, Any, Dict

from asme.core.init.config import Config
from asme.core.init.context import Context
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

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        for result in map(lambda factory: factory.can_build(config, context), self.factories):
            if result.type == CanBuildResultType.CAN_BUILD:
                return result

        return CanBuildResult(
            CanBuildResultType.INVALID_CONFIGURATION,
            f"None of the configured factories can handle the configuration section {'.'.join(config.base_path)}."
        )

    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        for factory in self.factories:
            can_build_result = factory.can_build(config, context)
            if can_build_result.type == CanBuildResultType.CAN_BUILD:
                return factory.build(config, context)

        raise Exception(f"No factory was able to build the configuration section {config.get_config([]).config}.")

    def is_required(self, context: Context) -> bool:
        return self.required

    def config_path(self) -> List[str]:
        return self._config_path

    def config_key(self) -> str:
        return self._config_key
