from typing import Type, Dict, Any

from attr import dataclass, field

from asme.core.init.object_factory import ObjectFactory
from asme.data.datasets.processors.processor import Processor


@dataclass
class ProcessorConfig:
    factory_cls: Type[ObjectFactory]
    factory_cls_args: Dict[str, Any] = field(factory=lambda: {})


REGISTERED_PREPROCESSORS = {}


def register_processor(key: str, config: ProcessorConfig,  overwrite: bool = False):
    if key in REGISTERED_PREPROCESSORS and not overwrite:
        raise KeyError(f"A processor with key '{key}' is already registered and overwrite was set to false.")
    REGISTERED_PREPROCESSORS[key] = config.factory_cls(**config.factory_cls_args)