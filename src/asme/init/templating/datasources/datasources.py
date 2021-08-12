from abc import abstractmethod
from enum import Enum
from typing import Dict, Any, List
from pathlib import Path

from asme.init.templating import TEMPLATES_CONFIG_KEY
from asme.init.templating.template_processor import TemplateProcessor


CONFIG_DATASOURCES_KEY = 'data_sources'

TARGET_EXTRACTOR_PROCESSOR_CONFIG = {
    'type': 'target_extractor'
}

POS_NEG_PROCESSOR_CONFIG = {
    'type': 'pos_neg'
}


class DatasetSplit(Enum):

    """
    all supported dataset splits
    """

    RATIO_SPLIT = 1
    LEAVE_ONE_OUT = 2


class Stage(Enum):
    """
    all stages for which we can apply templates for
    """
    TRAIN = "train"

    VALIDATION = "validation"

    TEST = "test"


class DataSourceTemplateProcessor(TemplateProcessor):

    """
    An abstract template processor that builds a data_sources definition (train, validation, test dataset)
    """

    def can_modify(self, config: Dict[str, Any]) -> bool:
        if TEMPLATES_CONFIG_KEY not in config:
            return False

        template_config = config.get(TEMPLATES_CONFIG_KEY)
        template_present = self._get_template_key() in template_config

        return template_present

    @abstractmethod
    def _get_template_key(self) -> str:
        pass

    def modify(self, config: Dict[str, Any]) -> Dict[str, Any]:
        template_config = config.get(TEMPLATES_CONFIG_KEY)
        data = template_config.pop(self._get_template_key())

        train_config = self._build_train_datasource(data)
        validation_config = self._build_validation_datasource(data)
        test_config = self._build_test_datasource(data)

        config[CONFIG_DATASOURCES_KEY] = {
            Stage.TRAIN.value: train_config,
            Stage.VALIDATION.value: validation_config,
            Stage.TEST.value: test_config
        }
        return config

    @abstractmethod
    def _build_train_datasource(self,
                                config: Dict[str, Any]
                                ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _build_validation_datasource(self,
                                     config: Dict[str, Any]
                                     ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _build_test_datasource(self,
                               config: Dict[str, Any]
                               ) -> Dict[str, Any]:
        pass


PARSER_ITEM_COLUMN_NAME = 'item_column_name'
PARSER_ITEM_SEPARATOR = 'item_separator'



def _get_prefix(config: Dict[str, Any],
                stage: Stage
                ) -> str:
    prefix = config.get('file_prefix', stage.value)
    prefix = config.get(f'{stage.value}_file_prefix', prefix)
    return prefix


class DatasetBuilder:

    @abstractmethod
    def can_build_dataset_definition(self, dataset_split_type: DatasetSplit) -> bool:
        pass

    @abstractmethod
    def build_dataset_definition(self, state: Stage, config: Dict[str, Any]) -> Dict[str, Any]:
        pass


class NextPositionDatasetBuilder(DatasetBuilder):
    def can_build_dataset_definition(self, dataset_split_type: DatasetSplit) -> bool:
        return dataset_split_type == DatasetSplit.RATIO_SPLIT

    def build_dataset_definition(self, stage: Stage, config: Dict[str, Any]) -> Dict[str, Any]:
        base_path = Path(config['path'])
        prefix = _get_prefix(config, stage)
        next_seq_step_type = config.get('next_seq_step_type', 'nextitem')
        stage_value = stage.value
        csv_file_path = base_path / f'{prefix}.{stage_value}.csv'
        csv_file_index_path = base_path / f'{prefix}.{stage_value}.session.idx'
        position_index_file_path = base_path / f'{prefix}.{stage_value}.{next_seq_step_type}.idx'
        return {
            'type': 'sequence_position',
            'csv_file': str(csv_file_path),
            'csv_file_index': str(csv_file_index_path),
            'nip_index_file': str(position_index_file_path)
        }


class SequenceDatasetRatioSplitBuilder(DatasetBuilder):

    def can_build_dataset_definition(self, dataset_split_type: DatasetSplit) -> bool:
        return dataset_split_type == DatasetSplit.RATIO_SPLIT

    def build_dataset_definition(self, stage: Stage, config: Dict[str, Any]) -> Dict[str, Any]:
        base_path = Path(config['path'])
        prefix = _get_prefix(config, stage)
        stage_value = stage.value
        csv_file_path = base_path / f'{prefix}.{stage_value}.csv'
        csv_file_index_path = base_path / f'{prefix}.{stage_value}.session.idx'
        return {
            'type': 'session',
            'csv_file': str(csv_file_path),
            'csv_file_index': str(csv_file_index_path)
        }


class ConditionalSequenceOrSequencePositionDatasetBuilder(DatasetBuilder):

    def __init__(self):
        super().__init__()
        self._sequence_dataset_builder = SequenceDatasetRatioSplitBuilder()
        self._sequence_position_dataset_builder = NextPositionDatasetBuilder()

    def can_build_dataset_definition(self, dataset_split_type: DatasetSplit) -> bool:
        return dataset_split_type == DatasetSplit.RATIO_SPLIT

    def build_dataset_definition(self, stage: Stage, config: Dict[str, Any]) -> Dict[str, Any]:
        if 'next_seq_step_type' in config:
            dataset_config = self._sequence_position_dataset_builder.build_dataset_definition(stage, config)
            return dataset_config

        return self._sequence_dataset_builder.build_dataset_definition(stage, config)


class LeaveOneOutNextPositionDatasetBuilder(DatasetBuilder):

    def can_build_dataset_definition(self, dataset_split_type: DatasetSplit) -> bool:
        return dataset_split_type == DatasetSplit.LEAVE_ONE_OUT

    def build_dataset_definition(self, stage: Stage, config: Dict[str, Any]) -> Dict[str, Any]:
        base_path = Path(config['path'])
        prefix = _get_prefix(config, stage)

        csv_file_path = base_path / f'{prefix}.csv'
        csv_file_index_path = base_path / f'{prefix}.session.idx'
        nip_index_file_path = base_path / 'loo' / f'{prefix}.{stage.value}.nextitem.idx'
        return {
            'type': 'sequence_position',
            'csv_file': str(csv_file_path),
            'csv_file_index': str(csv_file_index_path),
            'nip_index_file': str(nip_index_file_path)
        }


class LeaveOneOutSequenceWindowDatasetBuilder(DatasetBuilder):

    def __init__(self):
        pass

    def can_build_dataset_definition(self, dataset_split_type: DatasetSplit):
        return dataset_split_type == DatasetSplit.LEAVE_ONE_OUT

    def build_dataset_definition(self, stage: Stage, config: Dict[str, Any]) -> Dict[str, Any]:
        base_path = Path(config['path'])
        window_size = config["window_size"]
        prefix = _get_prefix(config, stage)
        index_file_path = f"{prefix}.{stage}"
        csv_file = base_path / f'{prefix}.csv'
        csv_file_index = base_path / f'{prefix}.session.idx'
        nip_index_file = base_path / 'loo' / f'{index_file_path}.slidingwindow.{window_size}.idx'
        return {
            'type': 'sequence_position',
            'csv_file': str(csv_file),
            'csv_file_index': str(csv_file_index),
            'nip_index_file': str(nip_index_file)
        }


class NextPositionWindowDatasetBuilder(DatasetBuilder):
    def __init__(self):
        pass

    def can_build_dataset_definition(self, dataset_split_type: DatasetSplit):
        return dataset_split_type == DatasetSplit.RATIO_SPLIT

    def build_dataset_definition(self, stage: Stage, config: Dict[str, Any]) -> Dict[str, Any]:
        base_path = Path(config['path'])
        window_size = config["window_size"]
        prefix = _get_prefix(config, stage)
        prefix = f"{prefix}.{stage.value}"
        csv_file = base_path / f'{prefix}.csv'
        csv_file_index = base_path / f'{prefix}.session.idx'
        nip_index_file = base_path / f'{prefix}.slidingwindow.{window_size}.idx'
        return {
            'type': 'sequence_position',
            'csv_file': str(csv_file),
            'csv_file_index': str(csv_file_index),
            'nip_index_file': str(nip_index_file)
        }


class LeaveOneOutSessionDatasetBuilder(DatasetBuilder):

    def can_build_dataset_definition(self, dataset_split_type: DatasetSplit):
        return dataset_split_type == DatasetSplit.LEAVE_ONE_OUT

    def build_dataset_definition(self, stage: Stage, config: Dict[str, Any]) -> Dict[str, Any]:
        base_path = Path(config['path'])
        prefix = _get_prefix(config, stage)
        index_file_path = f"{prefix}.{stage.value}"
        csv_file = base_path / f'{prefix}.csv'
        csv_file_index = base_path / f'{prefix}.session.idx'
        nip_index_file = base_path / 'loo' / f'{index_file_path}.loo.idx'
        return {
            'type': 'sequence_position',
            'csv_file': str(csv_file),
            'csv_file_index': str(csv_file_index),
            'nip_index_file': str(nip_index_file)
        }


def build_datasource(dataset_builders: List[DatasetBuilder],
                     config: Dict[str, Any],
                     stage: Stage,
                     additional_processors: List[Dict[str, Any]] = None
                     ) -> Dict[str, Any]:
    """
    builds a datasource config with the specified parser, processor,
    :param dataset_builders: the builders to use to build the dataset config
    :param config: the config of the template
    :param stage: the stage to build the datasource config for
    :param additional_processors:
    :return:
    """
    loader_config = config['loader']
    parser_config = config.get('parser')

    base_batch_size = loader_config.get('batch_size', 0)
    batch_size = loader_config.get(f'{stage.value}_batch_size', base_batch_size)
    shuffle = loader_config.get('shuffle', stage == Stage.TRAIN)

    processors = [
        {
            'type': 'tokenizer'
        }
    ]

    if additional_processors is not None:
        processors.extend(additional_processors)

    dataset_split_type = DatasetSplit[config.get('split_type', DatasetSplit.RATIO_SPLIT.name).upper()]

    def _build_dataset_config() -> Dict[str, Any]:
        for datasource_builder in dataset_builders:
            if datasource_builder.can_build_dataset_definition(dataset_split_type):
                return datasource_builder.build_dataset_definition(stage, config)
        raise ValueError('no datasource builder found')

    dataset_config = _build_dataset_config()
    dataset_config['processors'] = processors

    if parser_config is not None:
        dataset_config['parser'] = parser_config

    loader_config_dict = {
        'dataset': dataset_config,
        'batch_size': batch_size,
        'shuffle': shuffle
    }

    loader_config_dict = _transfer_properties(loader_config, loader_config_dict,
                                              ['max_seq_step_length', 'num_workers', 'dynamic_padding'])

    return {
        'loader': loader_config_dict
    }


def _transfer_properties(source_dict: Dict[str, Any],
                         target_dict: Dict[str, Any],
                         keys_to_transfer: List[str],
                         target_key_names: List[str] = None,
                         ) -> Dict[str, Any]:
    if target_key_names is None:
        target_key_names = keys_to_transfer
    for source_key, target_key in zip(keys_to_transfer, target_key_names):
        value = source_dict.get(source_key)
        if value is not None:
            target_dict[target_key] = value

    return target_dict
