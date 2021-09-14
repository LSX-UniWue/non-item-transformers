from dataclasses import dataclass
from typing import Dict, Any, List

from optuna import Trial


def key_path_to_str(key_path: List[str]) -> str:
    return '.'.join(key_path)


@dataclass
class ParameterDependencyInfo:
    """
    info class for saving information on the parameter dependency
    """
    dependency_type: str
    depends_on: str
    parameters: Dict[str, Any]


@dataclass
class ParameterInfo:

    key_path: List[str]
    suggest: str
    suggest_parameters: Dict[str, Any]
    dependency: ParameterDependencyInfo

    @property
    def parameter_key(self) -> str:
        return key_path_to_str(self.key_path)


class ParameterResolver:
    def resolve(self,
                parameter_config: ParameterInfo,
                resolved_values: Dict[str, Any]
                ) -> Any:
        raise NotImplementedError()

    def can_resolve(self,
                    key: str
                    ) -> bool:
        raise NotImplementedError


class OptunaParameterResolver(ParameterResolver):

    OPTUNA_KEY: str = "hyper_opt"

    def __init__(self, trial: Trial):
        self.trial = trial

    def _generate_hyperparameter(self, parameter_config: ParameterInfo) -> Any:
        function_name = 'suggest_' + parameter_config.suggest

        if not hasattr(self.trial, function_name):
            raise Exception(f"{function_name} is not a valid function name. Please refer to the documentation of the"
                            f" `Trial` object for valid names and arguments.")

        function_obj = getattr(self.trial, function_name)
        function_args = parameter_config.suggest_parameters.copy()
        function_args['name'] = parameter_config.parameter_key
        # TODO (AD) we could also validate that the provided arguments are sufficient
        # for calling the function and generate a meaningful error message

        return function_obj(**function_args)

    def resolve(self, parameter_config: ParameterInfo, resolved_values: Dict[str, Any]) -> Any:
        # XXX: do not generate the hyperparameter here, because we must skip the call if the conditions are not
        # satisfied when the dependency type is optimize_iff
        dependency = parameter_config.dependency
        if not dependency:
            return self._generate_hyperparameter(parameter_config)

        dependency_type = dependency.dependency_type
        dependent_value = resolved_values.get(dependency.depends_on)

        if 'optimize_iff' == dependency_type:
            conditions = dependency.parameters.get('conditions')
            satisfies_condition = True
            for condition in conditions:
                condition_type = condition.pop('type')
                if condition_type != 'equal':
                    raise KeyError(f'{condition_type} currently not supported')

                compare_value = condition.pop('compare_value')

                if compare_value != dependent_value:
                    satisfies_condition = False
                    break
            return self._generate_hyperparameter(parameter_config) if satisfies_condition else None

        value = self._generate_hyperparameter(parameter_config)
        dependency_func = {
            'multiply': lambda x, y: x * y
        }[dependency_type]

        return dependency_func(value, dependent_value)

    def can_resolve(self, key: str) -> bool:
        return key == self.OPTUNA_KEY


def _parse_dependency_info(dependency_info: Dict[str, Any]
                           ) -> ParameterDependencyInfo:
    depends_on = dependency_info.pop('on')
    dependency_type = dependency_info.pop('type')

    return ParameterDependencyInfo(dependency_type, depends_on, dependency_info)


def parse_parameter_dependency_info(current_key: List[str],
                                    value: Dict[str, Any]
                                    ) -> ParameterInfo:
    """
    parses parameter infos from the dict
    :param current_key:
    :param value:
    :return: the parameter info object
    """
    key_path = current_key[:-1]  # here we remove model hyper_opt at the end
    suggest_func = value['suggest']
    suggest_params = value['params']

    dependency = value.get('dependency', None)
    dependency_info = None

    if dependency is not None:
        dependency_info = _parse_dependency_info(dependency)

    return ParameterInfo(key_path, suggest_func, suggest_params, dependency_info)
