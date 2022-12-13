from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import Type

from asme.core.init.object_factory import ObjectFactory


@dataclass
class TemplateConfig:
    factory_cls: Type[ObjectFactory]
    factory_cls_args: Dict[str, Any] = field(default_factory=lambda: {})


REGISTERED_TEMPLATES = {}


def register_template(key: str, config: TemplateConfig,  overwrite: bool = False):
    if key in REGISTERED_TEMPLATES and not overwrite:
        raise KeyError(f"A template with key '{key}' is already registered and overwrite was set to false.")
    REGISTERED_TEMPLATES[key] = config.factory_cls(**config.factory_cls_args)
