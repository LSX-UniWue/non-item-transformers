from dataclasses import dataclass, field
from typing import Type, Dict, Any
from asme.core.init.object_factory import ObjectFactory


@dataclass
class EvaluationConfig:
    factory_cls: Type[ObjectFactory]
    factory_cls_args: Dict[str, Any] = field(default_factory=lambda: {})


REGISTERED_EVALUATORS = {}

def register_evaluator(key: str, config: EvaluationConfig, overwrite: bool = False):
    if key in REGISTERED_EVALUATORS and not overwrite:
        raise KeyError(f"A module with key '{key}' is already registered and overwrite was set to false.")
    REGISTERED_EVALUATORS[key] = config.factory_cls(**config.factory_cls_args)
