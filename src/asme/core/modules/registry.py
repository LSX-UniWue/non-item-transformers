from dataclasses import dataclass, field
from typing import Type, Dict, Any

from pytorch_lightning import LightningModule

from asme.core.init.object_factory import ObjectFactory


@dataclass
class ModuleConfig:
    factory_cls: Type[ObjectFactory]
    module_cls: Type[LightningModule]
    factory_cls_args: Dict[str, Any] = field(default_factory=lambda: {})


REGISTERED_MODULES = {}


def register_module(key: str, config: ModuleConfig,  overwrite: bool = False):
    if key in REGISTERED_MODULES and not overwrite:
        raise KeyError(f"A module with key '{key}' is already registered and overwrite was set to false.")
    REGISTERED_MODULES[key] = config.factory_cls(config.module_cls,**config.factory_cls_args)


