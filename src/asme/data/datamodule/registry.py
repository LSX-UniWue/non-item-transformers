from typing import Optional

from asme.data.datamodule.config import PreprocessingConfigProvider

DATASET_CONFIG_PROVIDERS = {}


def register_preprocessing_config_provider(name: str, config_provider: PreprocessingConfigProvider,
                                           overwrite: bool = False):
    """
    This function registers a preprocessing config provider with ASME. It can later be referenced by setting the
    dataset field in the configuration file to the name passed ot this function.

    :param name: The name that will later be used to reference this dataset via the configuration file.

    :param config_provider: The preprocessing config provider that encapsulates the generation of a preprocessing config
    and optionally holds default values for some parameters.

    :param overwrite: If True, a call to this method might overwrite an already existing preprocessing config provider
    with the same name. If the name already exists and overwrite is False, a KeyError is raised.
    """
    if name in DATASET_CONFIG_PROVIDERS and not overwrite:
        raise KeyError(f"A dataset config for key '{name}' is already registered and overwrite was set to false.")

    DATASET_CONFIG_PROVIDERS[name] = config_provider


def get_preprocessing_config_provider(name: str) -> Optional[PreprocessingConfigProvider]:
    return DATASET_CONFIG_PROVIDERS.get(name, None)
