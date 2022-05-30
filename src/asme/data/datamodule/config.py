import copy
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Callable, Any, Dict

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.data.datamodule.preprocessing.action import PreprocessingAction
from asme.data.datamodule.unpacker import Unpacker


@dataclass
class DatasetPreprocessingConfig:
    """
    Container for all information and actions necessary to convert some dataset into a format which is processable by
    ASME.
    """
    name: str
    url: Optional[str]
    location: Path
    unpacker: Optional[Unpacker] = None
    preprocessing_actions: List[PreprocessingAction] = field(default_factory=[])
    context: Context = Context()


@dataclass
class AsmeDataModuleConfig:
    """
    Container for all information that is necessary for the AsmeDatamodule to be able to correctly preprocess and load
    a dataset.
    """
    dataset: str
    cache_path: Optional[str]
    template: Optional[Config]
    data_sources: Optional[Config]
    preprocessing_config_values: Optional[Dict[str, Any]]
    dataset_preprocessing_config: Optional[DatasetPreprocessingConfig]
    force_regeneration: bool = False


class PreprocessingConfigProvider:
    """
        This class encapsulates a function that generates a DatasetPreprocessingConfig.
        It allows to specify default values for some of the parameters of the generating function.
    """

    def __init__(self, factory: Callable[..., DatasetPreprocessingConfig], **default_values):
        """
        :param factory: The function that actually generates the DatasetPreprocessingConfig. Parameters to this function
                        should by "integral" (i.e. string, int, list) types since they are directly filled with values
                        from the config file.

        :param default_values: A dictionary of default values to use for the factory function. If the functions contains
                               a parameter named "param" that does not have a default value specified in the function
                               signature and no value is provided to __call__, the value from this dictionary is used.
                               If no value is found, an exception is raised.
        """
        self.factory = factory
        self.default_values = default_values

    def __call__(self, **kwargs) -> DatasetPreprocessingConfig:
        """
        Invokes the encapsulated function with the values passed to this function. Missing values are replaced by ones
        passed in via the default_values dictionary. If no value is found for a non-optional parameter, an exception is
        raised function with the values passed to this function. Missing values are replaced by ones
        passed in via the default_values dictionary. If no value is found for a non-optional parameter, an exception is
        raised.
        """
        arguments = copy.deepcopy(self.default_values)
        arguments.update(kwargs)
        params = inspect.signature(self.factory).parameters
        actual_arguments = {}
        for name, param in params.items():
            if name not in arguments and param.default == inspect.Parameter.empty:
                raise KeyError(f"Parameter '{name}' is absent and no default value was provided. "
                               f"However, '{name}' is required by this PreprocessingConfigProvider.")

            if param.default == inspect.Parameter.empty:
                actual_arguments[name] = arguments[name]
        return self.factory(**arguments)


