import os
from typing import List, Union, Any, Dict

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories.util import check_config_keys_exist
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.core.utils.logging import get_logger
import importlib

logger = get_logger(__name__)


def _import_module(path: str):
    name, _ = os.path.splitext(path)
    spec = importlib.util.spec_from_loader(
        name,
        importlib.machinery.SourceFileLoader(name, path)
    )
    if spec is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ImportFactory(ObjectFactory):

    CONFIG_KEY = "imports"
    TOP_LEVEL_MODULE_NAME = "asme"
    DEFAULT_MODULES = [".data.datasets.config", ".core.modules.config"]

    def __init__(self):
        super().__init__()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        if check_config_keys_exist(config, [self.CONFIG_KEY]):
            for name, path in config.get(self.CONFIG_KEY).items():
                if not os.path.isfile(path):
                    return CanBuildResult(CanBuildResultType.INVALID_CONFIGURATION,
                                          f"The path specified for import '{name}' does not resolve to a valid file.")

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        # Load default modules
        for default_module in self.DEFAULT_MODULES:
            importlib.import_module(default_module, self.TOP_LEVEL_MODULE_NAME)

        # Load all extra modules specified in the config
        imports = config.get_or_default(self.CONFIG_KEY, {})
        if len(imports) > 0:
            logger.info(f"Loading {len(imports)} extra modules.")
            for name, path in imports.items():
                try:
                    module = _import_module(path)
                    if module is None:
                        logger.critical(f"Failed to load module '{name}' at '{path}'. Does the path exists?")
                        exit(1)
                    logger.info(f"Successfully loaded module '{name}' at '{path}'.")
                except Exception as e:
                    logger.critical(f"Failed to load module '{name}' at '{path}' due to the following exception:", exc_info=e)
                    exit(1)

        return None

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.CONFIG_KEY]

    def config_key(self) -> str:
        return self.CONFIG_KEY