from typing import List, Dict, Any

from asme.init.config import Config
from asme.init.templating.datasources.datasources import DatasetBuilder, Stage, DatasetSplit, _transfer_properties


def build_dataset_config(dataset_builders: List[DatasetBuilder], config: Config, stage: Stage, additional_processors: List[Dict[str, Any]] = None) -> Dict[str, Any]:

    if additional_processors is None:
        additional_processors = []

    processors = [{
        "type": "tokenizer"
    }] + additional_processors

    split = DatasetSplit[config.get("split").upper()]

    def _build_dataset_config() -> Dict[str, Any]:
        for datasource_builder in dataset_builders:
            if datasource_builder.can_build_dataset_definition(split):
                return datasource_builder.build_dataset_definition(stage, config.config)
        raise ValueError('no datasource builder found')

    datasource_config = _build_dataset_config()
    datasource_config["processors"] = processors

    return datasource_config


def build_default_loader_config(config: Config, stage: Stage, dataset_builders: List[DatasetBuilder], processors: List[Dict[str, Any]] = None) -> Config:

    dataset_config = build_dataset_config(dataset_builders, config, stage, processors)

    base_batch_size = config.get_or_default('batch_size', 8)
    batch_size = config.get_or_default(f'{stage.value}_batch_size', base_batch_size)
    shuffle = config.get_or_default('shuffle', stage == Stage.TRAIN)

    loader_config = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "dataset": dataset_config
    }

    loader_config = _transfer_properties(config.config, loader_config,
                                              ['max_seq_step_length', 'num_workers', 'dynamic_padding'])

    return Config(loader_config)