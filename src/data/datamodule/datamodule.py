import os

from pathlib import Path
from typing import Optional, List

import pytorch_lightning.core as pl
from torch.utils.data import DataLoader

from asme.init.context import Context
from asme.init.factories.data_sources.template_datasources import TemplateDataSourcesFactory
from asme.init.factories.data_sources.user_defined_datasources import UserDefinedDataSourcesFactory
from data.datamodule.config import AsmeDataModuleConfig
from datasets.dataset_pre_processing.utils import download_dataset


class AsmeDataModule(pl.LightningDataModule):

    PREPROCESSING_FINISHED_FLAG = ".PREPROCESSING_FINISHED"

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

        # Check whether we already preprocessed the dataset
        if self._check_finished_flag(ds_config.location):
            print("Found a finished flag in the target directory. Assuming the dataset is already preprocessed.")
        else:
            print("Preprocessing dataset:")

            if ds_config.url is not None:
                print(f"Downloading dataset...")
                dataset_file = download_dataset(ds_config.url, ds_config.location)
            else:
                print(f"No download URL specified, using local copy at '{ds_config.location}'")
                dataset_file = ds_config.location

            # If necessary, unpack the dataset
            if ds_config.unpacker is not None:
                print(f"Unpacking dataset...", end="")
                ds_config.unpacker(dataset_file)
                print("Done.")

            # Apply preprocessing steps
            for i, step in enumerate(ds_config.preprocessing_actions):
                print(f"Applying preprocessing step '{step.name()}' ({i+1}/{len(ds_config.preprocessing_actions)})...", end="")
                step.apply(ds_config.context)
                print("Done.")

            # Write finished flag
            (ds_config.location / self.PREPROCESSING_FINISHED_FLAG).touch()

    def setup(self, stage: Optional[str] = None):
        if len(msg := self._validate_config()) > 0:
            raise KeyError(f"Invalid config due to: {msg}.")

        if self.config.template is not None:
            factory = TemplateDataSourcesFactory("name")
            self._objects = factory.build(self.config.template, self.context)
        else:
            factory = UserDefinedDataSourcesFactory()
            self._objects = factory.build(self.config.data_sources, self.context)

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

    def _check_finished_flag(self, directory: Path) -> bool:
        return os.path.exists(directory / self.PREPROCESSING_FINISHED_FLAG)

    def _validate_config(self) -> List[str]:
        errors = []
        if self.config.template is not None and self.config.data_sources is not None:
            errors.append("Please specify one of 'template' or 'data_sources'")

        return errors