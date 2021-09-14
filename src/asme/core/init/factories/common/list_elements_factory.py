from typing import Any, List, Dict, Union

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class NamedListElementsFactory(ObjectFactory):
    """
        A factory that builds a list of named elements of the same type. The `config_key` and `config_path` are
        inferred from the `element_factory` object.
    """

    def __init__(self,
                 element_factory: ObjectFactory,
                 required: bool = True):
        """
        Creates a named element factory.

        :param element_factory: a factory that can build every named element.
        """
        super(NamedListElementsFactory, self).__init__()
        self.element_factory = element_factory
        self.required = required
        self._config_path = element_factory.config_path()
        self._config_key = element_factory.config_key()

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        """
        Checks whether all named elements in the config can be built using the `element_factory`.

        :param config: a configuration section containing multiple named elements.
        :param context: the current context.
        :return: a result.
        """
        actual_config = config.get([])
        if len(actual_config) == 0 and self.is_required(context):
            return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION, f"At least one element must be specified.")

        for name in actual_config.keys():
            element_config = config.get_config([name])
            factory_config = element_config.get_config(self.element_factory.config_path())
            can_build_result = self.element_factory.can_build(factory_config, context)

            if not can_build_result.type == CanBuildResultType.CAN_BUILD:
                return can_build_result

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        """
        Builds every element in the configuration using the given factory.

        :param config: a configuration section containing multiple named elements.
        :param context: the current context.

        :return: a dictionary with the element names pointing to the built objects.
        """
        elements = config.get([])

        result = {}
        for name in elements.keys():
            element_config = config.get_config([name])
            factory_config = element_config.get_config(self.element_factory.config_path())
            obj = self.element_factory.build(factory_config, context)

            result[name] = obj

        return result

    def is_required(self, context: Context) -> bool:
        return self.required

    def config_path(self) -> List[str]:
        return self._config_path

    def config_key(self) -> str:
        return self._config_key
