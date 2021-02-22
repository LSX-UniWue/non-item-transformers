from typing import Dict, List, Union, Any

from init.config import Config
from init.context import Context
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


# TODO (AD) rewrite dependencyTrait in a way that it automatically processes all dependencies where they key can be found
# in the configuration
class DependenciesFactory(ObjectFactory):
    """
    A trait that contains operations that can help manage dependent factories. The trait assumes that dependent
    factories are responsible for building a specific subsection of the configuration and that the subsection name
    matches the `config_path()` reported by the dependent factory.
    """

    def __init__(self,
                 dependencies: List[ObjectFactory],
                 config_key: str = "",
                 config_path: List[str] = [],
                 required: bool = True,
                 optional_based_on_path: bool = False):
        """
        Adds all dependencies with the assumption that they can be used to build the configuration subsection with their
        `config_key`.

        :param dependencies: a list of factories.
        :param config_key: the config key for this factory.
        :param config_path: the config path for this factory.
        :param required: whether this factory needs to be built.
        :param optional_based_on_path: the dependency should only be called if the path exists
        """
        super(DependenciesFactory, self).__init__()
        self._required = required
        self._config_path = config_path
        self._config_key = config_key
        self._optional_based_on_path = optional_based_on_path

        self._dependencies: Dict[str, ObjectFactory] = {}
        self.add_dependencies(dependencies)

    def add_dependencies(self, dependencies: List[ObjectFactory]):
        for factory in dependencies:
            self.add_dependency(factory)

    def add_dependency(self, dependency: ObjectFactory):
        key = dependency.config_key()
        if key in self._dependencies:
            raise Exception(f"A factory for path <{key}> is already registered.")
        self._dependencies[key] = dependency

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        for key, factory in self._dependencies.items():
            if not self._optional_based_on_path and not config.has_path(factory.config_path()) and factory.is_required(context):
                return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION, f"missing path <{factory.config_path}>")

            factory_config = config.get_config(factory.config_path())
            factory_can_build_result = factory.can_build(factory_config, context)
            if factory_can_build_result.type != CanBuildResultType.CAN_BUILD:
                return factory_can_build_result

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        result = {}
        for key, factory in self._dependencies.items():
            factory_config_path = factory.config_path()
            if self._optional_based_on_path and not config.has_path(factory_config_path):
                # we skip this dependency because the path is not present and the config allow this situation
                continue
            factory_config = config.get_config(factory_config_path)
            obj = factory.build(factory_config, context)
            result[key] = obj
        return result

    def is_required(self, context: Context) -> bool:
        return self._required

    def config_path(self) -> List[str]:
        return self._config_path

    def config_key(self) -> str:
        return self._config_key
