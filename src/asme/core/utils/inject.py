import functools
import importlib
import inspect
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List

from dataclasses_json import dataclass_json


from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories import GLOBAL_ASME_INJECTION_CONTEXT, BuildContext
from asme.core.init.factories.features.tokenizer_factory import get_tokenizer_key_for_voc
from asme.core.init.factories.modules.modules import GenericModelFactory
from asme.core.init.factories.modules.util import ParameterInfo, get_init_parameters, get_tokenizers_from_context, \
    get_dictionaries_from_context


# All injection annotations should inherit from this base class to ensure extendability for future use cases.
class Inject:
    pass


class InjectTokenizer(Inject):
    def __init__(self, feature_name: str):
        self.feature_name = feature_name


class InjectTokenizers(Inject):
    pass

class InjectDictionaries(Inject):
    pass


class InjectVocabularySize(Inject):
    def __init__(self, feature_name: str):
        self.feature_name = feature_name


class InjectModel(Inject):
    def __init__(self, model_cls, config_section_path: Optional[str] = None):
        self.model_cls = model_cls
        self.config_section_path = config_section_path


class InjectClass(Inject):
    def __init__(self, config_section_path: Optional[str] = None):
        """
        :param config_section_path: The path to the section which contains the data necessary to build the desired
            object.Nested obects in the config can be accessed by using ".".
        """
        self.config_section_path = config_section_path


class InjectList(Inject):
    def __init__(self, config_section_path: Optional[str] = None):
        """
        :param config_section_path: The path to the section which contains the data necessary to build the desired
            object.Nested obects in the config can be accessed by using ".".
        """
        self.config_section_path = config_section_path


class InjectDict(Inject):
    def __init__(self, config_section_path: Optional[str] = None):
        """
        :param config_section_path: The path to the section which contains the data necessary to build the desired
            object.Nested obects in the config can be accessed by using ".".
        """
        self.config_section_path = config_section_path


@dataclass_json
@dataclass
class InjectObjectConfig:
    cls_name: str
    module_name: Optional[str]
    parameters: Dict[str, Any] = field(default_factory=lambda: {})


def inject(**injects: Inject):
    """
    This annotation is supposed to be placed on the init function of any object built by an ASME factory. It allows to
    populate arbitrary parameters of the init function with values derived form the config or context. Note that this
    annotation requires the `ASME_GLOBAL_FACTORY_CONFIG`  and `ASME_GLOBAL_FACTORY_CONTEXT`  to be set to the
    config/context objects that are used by all factories. If you use the `ContainerFactory` this is done automatically.

    :param injects: A list of key value-pairs where the key represents the name of the parameter that should be
        populated via injection while the value is an object deriving from `Inject`. This instance the determines how
        injections are performed.
    """

    # We partially apply the actual injection function since you can not pass kwargs to the decorator directly
    return functools.partial(inject_partially_applied, injects=injects)


def inject_partially_applied(init, injects: Dict[str,Inject]):

    @functools.wraps(init)
    def new_init(self, *args, **kwargs):
        parameters = get_init_parameters(self)
        injectable_parameters = [
            ParameterInfo(
                param.parameter_name,
                param.parameter_type,
                param.default_value,
                injects[param.parameter_name]
            )
            for param in parameters if param.parameter_name in injects]
        parameter_dict = {name: value for name, value in kwargs.items() if name not in injects}
        # This resolves all injection directives
        _handle_injects(injectable_parameters, GLOBAL_ASME_INJECTION_CONTEXT, parameter_dict)
        init(self, *args, **parameter_dict)

    return new_init


def _check_config_paths_exists_or_throw(config: Config, config_section_path: List[str], param: ParameterInfo):
    if not config.has_path(config_section_path):
        if param.default_value == inspect._empty:
            raise KeyError(f"Parameter {param.parameter_name} should be injected from the "
                           f"configuration path '{'.'.join(config_section_path)}', but this path was not "
                           f"found in the config and no default value was provided.")


def _parse_inject_object_config(config: Config, config_section_path: List[str]) -> InjectObjectConfig:
    try:
        return InjectObjectConfig.from_dict(config.get_config(config_section_path).config)
    except Exception as e:
        raise ValueError(f"Failed to parse inject config from path '{'.'.join(config_section_path)}'.", e)


def _build(config: Config, path: List[str], current_obj_name: str = ""):
    if not config.has_path(path):
        raise KeyError(f"Failed to build {current_obj_name if len(current_obj_name) > 0 else 'object'}"
                       f"with configuration path '{'.'.join(path)}' since this path does not"
                       f"exist in the config.")

    obj_config = _parse_inject_object_config(config, path)
    parameter_dict = {}
    for k, v in obj_config.parameters.items():
        # Recursively instantiate parameters
        if isinstance(v, dict) and "cls_name" in v:
            sub_path = path + ["parameters", k]
            name = current_obj_name + f".{k}" if len(current_obj_name) > 0 else k
            parameter_dict[k] = _build(config, sub_path, current_obj_name=name)
        else:
            parameter_dict[k] = v

    # Import module and build object
    obj_module = importlib.import_module(obj_config.module_name)
    obj_class = obj_module.__getattribute__(obj_config.cls_name)
    return obj_class(**parameter_dict)


def _handle_injects(injectable_parameters: List[ParameterInfo],
                    injection_context: BuildContext,
                    parameter_dict: Dict[str, Any]):
    """
    This function tries to inject a value into the parameter_dict for every parameter given by injectable parameters.

    :param injectable_parameters: A list of parameter info that are injectable, i.e. parameter infos
    p where p.is_injectable() == True
    :param injection_context: the injection context object
    :param parameter_dict: A dictionary of parameter names and values that should be populated by values gathered from
    the context or the config.
    """

    def _determine_config_path(config_path: Optional[str], parameter: ParameterInfo) -> List[str]:
        if config_path is None:
            return [parameter.parameter_name]
        else:
            return config_path.split(".")

    def _format_config_section_path(config_path: List[str], config: Config) -> str:
        return ".".join(config.base_path + config_path)

    config = injection_context.get_current_config_section()
    context = injection_context.get_context()

    for injectable_parameter in injectable_parameters:
        inject = injectable_parameter.inject_instance

        if isinstance(inject, InjectTokenizer):
            tokenizer_to_use = inject.feature_name
            tokenizer = context.get(get_tokenizer_key_for_voc(tokenizer_to_use))

            if tokenizer is None:
                if injectable_parameter.default_value == inspect._empty:
                    raise KeyError(f'No tokenizer with id "{tokenizer_to_use}" configured and no default value set.')
                else:
                    tokenizer = injectable_parameter.default_value
            parameter_dict[injectable_parameter.parameter_name] = tokenizer
        elif isinstance(inject, InjectTokenizers):
            tokenizers = get_tokenizers_from_context(context)
            parameter_dict[injectable_parameter.parameter_name] = tokenizers
        elif isinstance(inject, InjectDictionaries):
            dictionaries = get_dictionaries_from_context(context)
            parameter_dict[injectable_parameter.parameter_name] = dictionaries
        elif isinstance(inject, InjectVocabularySize):
            tokenizer_to_use = inject.feature_name
            tokenizer = context.get(get_tokenizer_key_for_voc(tokenizer_to_use))
            if tokenizer is None:
                raise KeyError(f"No tokenizer with id {tokenizer_to_use} configured. "
                               f"Can not inject vocabulary size into parameter {injectable_parameter.parameter_name}.")
            else:
                parameter_dict[injectable_parameter.parameter_name] = len(tokenizer)
        elif isinstance(inject, InjectModel):
            # We have to handle InjectModel separately from InjectClass since it requires the use of a ModelFactory.
            config_section_path = _determine_config_path(inject.config_section_path, injectable_parameter)

            _check_config_paths_exists_or_throw(config, config_section_path, injectable_parameter)
            factory = GenericModelFactory(inject.model_cls)
            model_config = config.get_config(config_section_path)
            model = factory.build(model_config, context)
            parameter_dict[injectable_parameter.parameter_name] = model
        elif isinstance(inject, InjectClass):
            config_section_path = _determine_config_path(inject.config_section_path, injectable_parameter)
            _check_config_paths_exists_or_throw(config, config_section_path, injectable_parameter)
            parameter_dict[injectable_parameter.parameter_name] = _build(config,
                                                                         config_section_path,
                                                                         injectable_parameter.parameter_name)
        elif isinstance(inject, InjectList):
            config_section_path = _determine_config_path(inject.config_section_path, injectable_parameter)
            _check_config_paths_exists_or_throw(config, config_section_path, injectable_parameter)
            obj_config = config.get(config_section_path)
            if not isinstance(obj_config, list):
                raise ValueError(f"Parameter '{injectable_parameter.parameter_name}' was annotated with 'InjectList'"
                                 f" but the provided config section "
                                 f"'{_format_config_section_path(config_section_path, config)}' is not a list.")

            instances = []
            for i, param in enumerate(obj_config):
                if isinstance(param, dict) and "cls_name" in param:
                    # Recursively build sub objects
                    config_obj = Config(param)
                    instances += [_build(config_obj, [], f"{injectable_parameter.parameter_name}.[{i}]")]
                else:
                    instances += [param]

            parameter_dict[injectable_parameter.parameter_name] = instances
        elif isinstance(inject, InjectDict):
            config_section_path = _determine_config_path(inject.config_section_path, injectable_parameter)
            _check_config_paths_exists_or_throw(config, config_section_path, injectable_parameter)
            obj_config = config.get(config_section_path)
            if not isinstance(obj_config, dict):
                raise ValueError(
                    f"Parameter '{injectable_parameter.parameter_name}' was annotated with 'InjectDict' but"
                    f" the provided config section "
                    f"'{_format_config_section_path(config_section_path, config)}' is not a dictionary.")
            instances = {}
            for key, param in obj_config.items():
                if isinstance(param, dict) and "cls_name" in param:
                    # Recursively build sub objects
                    config_obj = Config(param)
                    instances[key] = _build(config_obj, [], f"{injectable_parameter.parameter_name}.{key}")
                else:
                    instances[key] = param

            parameter_dict[injectable_parameter.parameter_name] = instances
        else:
            # We failed ot handle an inject case!
            raise NotImplementedError(f"Inject instances of type {type(inject)} are not handled yet!")

