import abc
from typing import Any, List, Optional, Union, Dict
from enum import Enum

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories import BuildContext


class CanBuildResultType(Enum):
    CAN_BUILD = 1
    MISSING_DEPENDENCY = 2
    MISSING_CONFIGURATION = 3
    INVALID_CONFIGURATION = 4


class CanBuildResult:
    """
    An object that gives detailed information about the build status of a factory.
    """
    def __init__(self,
                 type: CanBuildResultType,
                 message: Optional[str] = None
                 ):
        """
        The constructor.

        :param type: a type.
        :param message: a message that further explains the result type.
        """
        self.type = type
        self.message = message


class ObjectFactory:

    def __init__(self):
        super(ObjectFactory, self).__init__()  # make this constructor cooperative so we can have multiple traits

    """
    Interface for factories that can participate in building objects from the configuration file. Factories operate only
    on the part of the configuration file, marked by their `config_path()` method and can build hierarchies to perfrom
    object initialization.
    """
    @abc.abstractmethod
    def can_build(self,
                  build_context: BuildContext
                  ) -> CanBuildResult:
        """
        Checks if the factory can build its object, based on the configuration and current context.
        This method is expected to raise an Exception if the constraints can never be satisfied, e.g. some configuration
        is missing.

        :param build_context: the build context.

        :return: :code a result.
        """
        pass

    @abc.abstractmethod
    def build(self,
              build_context: BuildContext
              ) -> Union[Any, Dict[str, Any], List[Any]]:
        """
        Builds the object.

        Note: By design, the caller is responsible for adding the object to the context, if so desired.

       :param build_context: the build context.

        :return the object that has been build, can be a dictionary if the factory builds several objects.

        """
        pass

    @abc.abstractmethod
    def is_required(self, build_context: BuildContext) -> bool:
        """
        Tells whether this factory's object needs to be build.
              :param build_context: the build context.


        :return: :code true if this factory must be called, :code false otherwise.
        """
        pass

    @abc.abstractmethod
    def config_path(self) -> List[str]:
        """
        Gets configuration path this factory can build.

        :return: a path.
        """
        pass

    @abc.abstractmethod
    def config_key(self) -> str:
        """
        Gets a key for this factory. This is used to identify the object, e.g. if added as a dependency.
        :return: a key.
        """
        pass
