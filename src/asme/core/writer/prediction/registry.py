from dataclasses import dataclass, field
from typing import Type, Dict, Any
from asme.core.init.object_factory import ObjectFactory


@dataclass
class WriterConfig:
    factory_cls: Type[ObjectFactory]
    factory_cls_args: Dict[str, Any] = field(default_factory=lambda: {})


REGISTERED_WRITERS = {}

def register_writer(key: str, config: WriterConfig, overwrite: bool = False):
    if key in REGISTERED_WRITERS and not overwrite:
        raise KeyError(f"A writer with key '{key}' is already registered and overwrite was set to false.")
    REGISTERED_WRITERS[key] = config.factory_cls(**config.factory_cls_args)
