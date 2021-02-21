from typing import List, Union, Any, Dict

from init.config import Config
from init.context import Context
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class SelectFromFactory(ObjectFactory):
    """
    Selects the appropriate factory to handle a configuration section from a list of possible candidates. If multiple
    factories are eligible to handle the configuration section, the first one as supplied during instantiation is used.
    """

    def __init__(self, key: str, required: bool, factories: List[ObjectFactory]):
        super(SelectFromFactory, self).__init__()
        self.key = key
        self.required = required
        self.factories = factories

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        for result in map(lambda factory: factory.can_build(config, context), self.factories):
            if result.type == CanBuildResultType.CAN_BUILD:
                return result

        return CanBuildResult(
            CanBuildResultType.INVALID_CONFIGURATION,
            f"None of the configured factories can handle the configuration section {config.get_config([])}"
        )

    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        for factory in self.factories:
            can_build_result = factory.can_build(config, context)
            if can_build_result.type == CanBuildResultType.CAN_BUILD:
                return factory.build(config, context)

        raise Exception(f"No factory was able to build the configuration section {config.get_config([])}")

    def is_required(self, context: Context) -> bool:
        return self.required

    def config_path(self) -> List[str]:
        return [self.key]

    def config_key(self) -> str:
        return self.key
