import distutils
from typing import List, Union, Any, Dict

from asme.init.config import Config
from asme.init.context import Context
from asme.init.factories.common.conditional_based_factory import ConditionalFactory
from asme.init.factories.util import check_config_keys_exist
from asme.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from data.datamodule.datamodule import AsmeDataModule
from data.datamodule.config import get_preprocessing_config_provider, AsmeDataModuleConfig


class DataModuleFactory(ObjectFactory):

    CONFIG_KEY = "datamodule"
    REQUIRED_CONFIG_KEYS = ["dataset"]

    def __init__(self):
        super().__init__()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        if not check_config_keys_exist(config, self.REQUIRED_CONFIG_KEYS):
            return CanBuildResult(
                CanBuildResultType.MISSING_CONFIGURATION,
                f"Could not find all required keys ({self.REQUIRED_CONFIG_KEYS}) in config."
            )

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> AsmeDataModule:
        # DO NOT TOUCH THIS
        import data.datasets.config

        dataset_name = config.get("dataset")
        force_regeneration = bool(distutils.util.strtobool(config.get_or_default("force_regeneration", "False")))
        dataset_preprocessing_config_provider = get_preprocessing_config_provider(dataset_name)
        if dataset_preprocessing_config_provider is None:
            print(f"No dataset registered for key '{dataset_name}'. No preprocessing will be applied.")
            dataset_preprocessing_config = None
        else:
            preprocessing_config_values = {**config.get_config(["preprocessing"]).config} if config.has_path(["preprocessing"]) else {}
            dataset_preprocessing_config = dataset_preprocessing_config_provider(**preprocessing_config_values)

        if config.has_path("template") and config.has_path("data_sources"):
            raise KeyError("Found both keys 'template' and 'data_sources'. Please specify exactly one.")

        cache_path = config.get("cache_path")
        data_sources_config = config.get_config(["data_sources"]) if config.has_path("data_sources") else None
        template_config = config.get_config(["template"]) if config.has_path("template") else None

        datamodule_config = AsmeDataModuleConfig(dataset_name, cache_path, template_config, data_sources_config,
                                                 dataset_preprocessing_config, force_regeneration)
        return AsmeDataModule(datamodule_config, context)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.CONFIG_KEY]

    def config_key(self) -> str:
        return self.CONFIG_KEY