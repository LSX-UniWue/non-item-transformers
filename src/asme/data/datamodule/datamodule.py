import os
import shutil

from typing import Optional, List, Union

import pytorch_lightning.core as pl
from torch.utils.data import DataLoader

from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.factories.data_sources.template_datasources import TemplateDataSourcesFactory
from asme.core.init.factories.data_sources.user_defined_datasources import UserDefinedDataSourcesFactory
from asme.core.init.templating.datasources.datasources import DatasetSplit
from asme.core.utils.logging import get_logger
from asme.data import BASE_DATASET_PATH_CONTEXT_KEY, CURRENT_SPLIT_PATH_CONTEXT_KEY, DATASET_PREFIX_CONTEXT_KEY, \
    RATIO_SPLIT_PATH_CONTEXT_KEY, LOO_SPLIT_PATH_CONTEXT_KEY, LPO_SPLIT_PATH_CONTEXT_KEY
from asme.data.datamodule.config import AsmeDataModuleConfig
from asme.data.datamodule.util import download_dataset

logger = get_logger(__name__)


class AsmeDataModule(pl.LightningDataModule):

    def __init__(self, config: AsmeDataModuleConfig, context: Context = Context()):
        super().__init__()
        self.config = config
        self.context = context

        self._objects = {}
        self._has_setup = False

    @property
    def has_setup(self):
        return self._has_setup

    def prepare_data(self):
        ds_config = self.config.dataset_preprocessing_config

        logger.info("Preprocessing dataset:")

        if ds_config.url is not None:
            logger.info(
                f"Downloading dataset {self.config.dataset} from {self.config.dataset_preprocessing_config.url}.")
            dataset_file = download_dataset(ds_config.url, ds_config.location)
        else:
            logger.info(f"No download URL specified, using local copy at '{ds_config.location}'.")
            dataset_file = ds_config.location

        # If necessary, unpack the dataset
        if ds_config.unpacker is not None:
            logger.info(f"Unpacking dataset.")
            ds_config.unpacker(dataset_file, force_unpack=self.config.force_regeneration)

        # Apply preprocessing steps
        for i, step in enumerate(ds_config.preprocessing_actions):
            logger.info(
                f"Applying preprocessing step '{step.name()}' ({i + 1}/{len(ds_config.preprocessing_actions)})")
            if step.dry_run_available(ds_config.context) and not self.config.force_regeneration:
                logger.info(f"Skipping this step since dry run is available.")
            step.apply(ds_config.context, self.config.force_regeneration)

        # Populate context with the dataset path
        self.context.set(BASE_DATASET_PATH_CONTEXT_KEY, self.config.dataset_preprocessing_config.location)
        self._set_split_path_in_context()
        # Also put the prefix into the context
        self._populate_config_and_context_with_prefix()

    def setup(self, stage: Optional[str] = None):
        msg = self._validate_config()
        if len(msg) > 0:
            raise KeyError(f"Invalid config: {msg}.")

        if self.config.cache_path is not None:
            # Copy the dataset
            ds_location = self.config.dataset_preprocessing_config.location
            logger.info(f"Caching dataset from '{ds_location}' to '{self.config.cache_path}'.")
            shutil.copytree(ds_location, self.config.cache_path, dirs_exist_ok=True, copy_function=shutil.copy)

            # adjust the context values such that the factories infer correct paths
            self.context.set(BASE_DATASET_PATH_CONTEXT_KEY, self.config.cache_path, overwrite=True)
            self._adjust_split_path_for_caching(CURRENT_SPLIT_PATH_CONTEXT_KEY)

        # Build the datasources depending on what the user specified
        if self.config.template is not None:
            factory = TemplateDataSourcesFactory("name")

            self._objects = factory.build(BuildContext(self.config.template, self.context))
        else:
            factory = UserDefinedDataSourcesFactory()
            self._objects = factory.build(BuildContext(self.config.data_sources, self.context))

        self._has_setup = True

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader("validation")

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader("test")

    def _get_dataloader(self, name: str):
        if not self.has_setup:
            self.setup()
        return self._objects[name]

    def _determine_split(self) -> Optional[DatasetSplit]:
        if self.config.data_sources is not None:
            split = self.config.data_sources.get("split")
        else:
            split = self.config.template.get("split")

        if split is None:
            return None
        else:
            return DatasetSplit[split.upper()]

    def _set_split_path_in_context(self):
        split = self._determine_split()
        if split == DatasetSplit.RATIO_SPLIT:
            split_path = self.config.dataset_preprocessing_config.context.get(RATIO_SPLIT_PATH_CONTEXT_KEY)
        elif split == DatasetSplit.LEAVE_ONE_OUT:
            split_path = self.config.dataset_preprocessing_config.context.get(LOO_SPLIT_PATH_CONTEXT_KEY)
        elif split == DatasetSplit.LEAVE_PERCENTAGE_OUT:
            split_path = self.config.dataset_preprocessing_config.context.get(LPO_SPLIT_PATH_CONTEXT_KEY)
        else:
            raise ValueError(f"Unkown split type: {split}.")

        self.context.set(CURRENT_SPLIT_PATH_CONTEXT_KEY, split_path)

    def _adjust_split_path_for_caching(self, key: Union[str, List[str]]):
        """
        Adjusts the split path stored in the context with the specified key to point to the cache directory instead.
        """
        current_value = self.context.get(key)
        split_dir = os.path.split(current_value)[-1]
        self.context.set(key, os.path.join(self.config.cache_path, split_dir), overwrite=True)

    def _populate_config_and_context_with_prefix(self):
        """
        Checks if the config includes a "file_prefix" param and propagates it to the context.
        If there is no "file_prefix" specified, it uses the dataset name as a prefix for both the config and context.
        """
        if self.config.template is not None:
            if self.config.template.has_path("file_prefix"):
                self.context.set(DATASET_PREFIX_CONTEXT_KEY, self.config.template.get("file_prefix"))
            else:
                self.config.template.set("file_prefix", self.config.dataset)
        elif self.config.data_sources is not None:
            if self.config.data_sources.has_path("file_prefix"):
                self.context.set(DATASET_PREFIX_CONTEXT_KEY, self.config.data_sources.get("file_prefix"))
            else:
                self.config.data_sources.set("file_prefix", self.config.dataset)

        if not self.context.has_path(DATASET_PREFIX_CONTEXT_KEY):
            self.context.set(DATASET_PREFIX_CONTEXT_KEY, self.config.dataset)

    def _validate_config(self) -> List[str]:
        errors = []
        if self.config.template is not None and self.config.data_sources is not None:
            errors.append("Please specify one of 'template' or 'data_sources'")

        if self._determine_split() is None:
            errors.append("Please specify a split type via the 'split' attribute. (Either 'leave_one_out' or 'ratio').")

        return errors
