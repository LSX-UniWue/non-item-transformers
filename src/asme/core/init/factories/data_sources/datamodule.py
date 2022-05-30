import distutils
from typing import List

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.factories.util import check_config_keys_exist
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.data.datamodule.datamodule import AsmeDataModule
from asme.data.datamodule.config import AsmeDataModuleConfig
from asme.data.datamodule.registry import get_preprocessing_config_provider


class DataModuleFactory(ObjectFactory):

    CONFIG_KEY = "datamodule"
    REQUIRED_CONFIG_KEYS = ["dataset"]

    def __init__(self):
        super().__init__()

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        if not check_config_keys_exist(build_context.get_current_config_section(), self.REQUIRED_CONFIG_KEYS):
            return CanBuildResult(
                CanBuildResultType.MISSING_CONFIGURATION,
                f"Could not find all required keys ({self.REQUIRED_CONFIG_KEYS}) in config."
            )

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, build_context: BuildContext) -> AsmeDataModule:
        config = build_context.get_current_config_section()
        dataset_name = config.get("dataset")
        force_regeneration = bool(distutils.util.strtobool(config.get_or_default("force_regeneration", "False")))
        dataset_preprocessing_config_provider = get_preprocessing_config_provider(dataset_name)
        if dataset_preprocessing_config_provider is None:
            print(f"No dataset registered for key '{dataset_name}'. No preprocessing will be applied.")
            preprocessing_config_values = None
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
                                                 preprocessing_config_values, dataset_preprocessing_config,
                                                 force_regeneration)
        datamodule = AsmeDataModule(datamodule_config, build_context.get_context())
        return datamodule

    def is_required(self, build_context: BuildContext) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.CONFIG_KEY]

    def config_key(self) -> str:
        return self.CONFIG_KEY