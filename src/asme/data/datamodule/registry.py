from typing import Optional

from asme.data.datamodule.config import PreprocessingConfigProvider

DATASET_CONFIG_PROVIDERS = {}


def register_preprocessing_config_provider(name: str, config_provider: PreprocessingConfigProvider, overwrite: bool = False):
    if name in DATASET_CONFIG_PROVIDERS and not overwrite:
        raise KeyError(f"A dataset config for key '{name}' is already registered and overwrite was set to false.")

    DATASET_CONFIG_PROVIDERS[name] = config_provider


def get_preprocessing_config_provider(name: str) -> Optional[PreprocessingConfigProvider]:
    return DATASET_CONFIG_PROVIDERS.get(name, None)