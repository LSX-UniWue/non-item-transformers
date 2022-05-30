from typing import Dict, List, Union, Any


from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.factories.util import can_build_with_subsection, build_with_subsection
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


# TODO (AD) rewrite dependencyTrait in a way that it automatically processes all dependencies where they key can be found
# in the configuration
class DependenciesFactory(ObjectFactory):

    def __init__(self,
                 dependencies: List[ObjectFactory],
                 config_key: str = "",
                 config_path: List[str] = None,
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
        super().__init__()
        if config_path is None:
            config_path = []
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

    def get_dependency_keys(self) -> List[str]:
        """
        Retrieves the keys for all registered dependencies.

        :return: a list with keys for all dependencies.
        """
        return [key for key, _ in self._dependencies.items()]

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        config = build_context.get_current_config_section()
        context = build_context.get_context()

        for key, factory in self._dependencies.items():
            if not self._optional_based_on_path and not config.has_path(factory.config_path()) and factory.is_required(build_context):
                return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION, f"missing path <{'.'.join(factory.config_path())}>")

            factory_can_build_result =can_build_with_subsection(factory, build_context)
            if factory_can_build_result.type != CanBuildResultType.CAN_BUILD:
                return factory_can_build_result

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, build_context: BuildContext) -> Union[Any, Dict[str, Any], List[Any]]:
        result = {}
        for key, factory in self._dependencies.items():
            factory_config_path = factory.config_path()

            if self._optional_based_on_path and not build_context.get_current_config_section().has_path(factory_config_path):
                # we skip this dependency because the path is not present and the config allow this situation
                continue
            result[key] = build_with_subsection(factory, build_context)
        return result

    def is_required(self, build_context: BuildContext) -> bool:
        return self._required

    def config_path(self) -> List[str]:
        return self._config_path

    def config_key(self) -> str:
        return self._config_key
