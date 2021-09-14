from abc import abstractmethod
from typing import Any, List, Union, Dict

from torch.utils.data import DataLoader

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories.common.conditional_based_factory import ConditionalFactory
from asme.core.init.factories.data_sources.common import build_default_loader_config, set_path_based_on_split
from asme.core.init.factories.data_sources.datasets.processor.processors import FIXED_SEQUENCE_LENGTH_PROCESSOR_KEY
from asme.core.init.factories.data_sources.loader import LoaderFactory
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.core.init.templating.datasources.datasources import Stage, SequenceDatasetRatioSplitBuilder, \
    LeaveOneOutSessionDatasetBuilder, \
    NextPositionDatasetBuilder, TARGET_EXTRACTOR_PROCESSOR_CONFIG, LeaveOneOutNextPositionDatasetBuilder, \
    ConditionalSequenceOrSequencePositionDatasetBuilder, POS_NEG_PROCESSOR_CONFIG, NextPositionWindowDatasetBuilder, \
    LeaveOneOutSequenceWindowDatasetBuilder, DatasetSplit
from asme.data import CURRENT_SPLIT_PATH_CONTEXT_KEY


class TemplateDataSourcesFactory(ObjectFactory):

    def __init__(self, key: str):
        super().__init__()
        self._key = key
        self._factory = ConditionalFactory(key, {
            "masked": MaskTemplateDataSourcesFactory(),
            "pos_neg": PositiveNegativeTemplateDataSourcesFactory(),
            "next_sequence_step": NextSequenceStepTemplateDataSourcesFactory(),
            "par_pos_neg": ParameterizedPositiveNegativeTemplateDataSourcesFactory(),
            "plain": PlainTrainingTemplateDataSourcesFactory(),
            "sliding_window": SlidingWindowTemplateDataSourcesFactory()
        })

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return self._factory.can_build(config, context)

    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        # If no dataset path was specified, try to the use the one provided by the datamodule
        split = DatasetSplit[config.get("split").upper()]
        set_path_based_on_split(config, context, split)
        return self._factory.build(config, context)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return ["template"]

    def config_key(self) -> str:
        return "template"


class BaseTemplateDataSourcesFactory(ObjectFactory):

    def __init__(self):
        super().__init__()
        self._loader_factory = LoaderFactory()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> Dict[str, DataLoader]:
        return {
            "train": self._build_train_datasource(config, context),
            "validation": self._build_validation_datasource(config, context),
            "test": self._build_test_datasource(config, context)
        }

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return ""

    def _build_datasource(self, config: Config, context: Context) -> DataLoader:
        return self._loader_factory.build(config, context)

    @abstractmethod
    def _build_train_datasource(self, config: Config, context: Context) -> DataLoader:
        pass

    @abstractmethod
    def _build_validation_datasource(self, config: Config, context: Context) -> DataLoader:
        pass

    @abstractmethod
    def _build_test_datasource(self, config: Config, context: Context) -> DataLoader:
        pass


class MaskTemplateDataSourcesFactory(BaseTemplateDataSourcesFactory):

    TRAIN_DATASET_BUILDERS = [SequenceDatasetRatioSplitBuilder(), LeaveOneOutSessionDatasetBuilder()]
    TEST_VALID_DATASET_BUILDERS = [NextPositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder()]

    def __init__(self):
        super().__init__()

    def _build_train_datasource(self, config: Config, context: Context) -> DataLoader:
        mask_probability = config.get_or_default('mask_probability', 0.2)
        only_last_item_mask_prob = config.get_or_default('only_last_item_mask_prob', 0.1)
        cloze_processor = {
            'type': "cloze",
            'mask_probability': mask_probability,
            'only_last_item_mask_prob': only_last_item_mask_prob,
        }
        loader_config = build_default_loader_config(config, Stage.TRAIN, self.TRAIN_DATASET_BUILDERS, [cloze_processor])
        return self._build_datasource(loader_config, context)

    def _build_validation_datasource(self, config: Config, context: Context) -> DataLoader:
        mask_last_item_processor = {
            'type': 'last_item_mask'
        }
        loader_config = build_default_loader_config(config, Stage.VALIDATION, self.TEST_VALID_DATASET_BUILDERS,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG, mask_last_item_processor])
        return self._build_datasource(loader_config, context)

    def _build_test_datasource(self, config: Config, context: Context) -> DataLoader:
        mask_last_item_processor = {
            'type': 'last_item_mask'
        }
        loader_config = build_default_loader_config(config, Stage.TEST, self.TEST_VALID_DATASET_BUILDERS,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG, mask_last_item_processor])
        return self._build_datasource(loader_config, context)


class NextSequenceStepTemplateDataSourcesFactory(BaseTemplateDataSourcesFactory):

    TRAIN_DATASET_BUILDERS = [NextPositionDatasetBuilder(), LeaveOneOutNextPositionDatasetBuilder()]
    TEST_VALID_DATASET_BUILDERS = [NextPositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder()]

    def __init__(self):
        super().__init__()

    def _build_train_datasource(self, config: Config, context: Context) -> DataLoader:
        loader_config = build_default_loader_config(config, Stage.TRAIN, self.TRAIN_DATASET_BUILDERS,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, context)

    def _build_validation_datasource(self, config: Config, context: Context) -> DataLoader:
        loader_config = build_default_loader_config(config, Stage.VALIDATION, self.TEST_VALID_DATASET_BUILDERS,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, context)

    def _build_test_datasource(self, config: Config, context: Context) -> DataLoader:
        loader_config = build_default_loader_config(config, Stage.TEST, self.TEST_VALID_DATASET_BUILDERS,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, context)


class PositiveNegativeTemplateDataSourcesFactory(BaseTemplateDataSourcesFactory):

    TRAIN_DATASET_BUILDERS = [ConditionalSequenceOrSequencePositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder()]
    TEST_VALID_DATASET_BUILDERS = [NextPositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder()]

    def __init__(self):
        super().__init__()

    def _build_train_datasource(self, config: Config, context: Context) -> DataLoader:
        loader_config = build_default_loader_config(config,
                                                    Stage.TRAIN,
                                                    self.TRAIN_DATASET_BUILDERS,
                                                    [POS_NEG_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, context)

    def _build_validation_datasource(self, config: Config, context: Context) -> DataLoader:
        loader_config = build_default_loader_config(config,
                                                    Stage.VALIDATION,
                                                    self.TEST_VALID_DATASET_BUILDERS,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, context)

    def _build_test_datasource(self, config: Config, context: Context) -> DataLoader:
        loader_config = build_default_loader_config(config,
                                                    Stage.TEST,
                                                    self.TEST_VALID_DATASET_BUILDERS,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, context)


class PlainTrainingTemplateDataSourcesFactory(BaseTemplateDataSourcesFactory):

    DATASET_BUILDER_TRAINING = [SequenceDatasetRatioSplitBuilder()]
    DATASET_BUILDERS_VALIDATION_AND_TEST = [NextPositionDatasetBuilder()]

    def _build_train_datasource(self, config: Config, context: Context) -> DataLoader:
        loader_config = build_default_loader_config(config, Stage.TRAIN, self.DATASET_BUILDER_TRAINING)
        return self._build_datasource(loader_config, context)

    def _build_validation_datasource(self, config: Config, context: Context) -> DataLoader:
        loader_config = build_default_loader_config(config,
                                                    Stage.VALIDATION,
                                                    self.DATASET_BUILDERS_VALIDATION_AND_TEST,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, context)

    def _build_test_datasource(self, config: Config, context: Context) -> DataLoader:
        loader_config = build_default_loader_config(config,
                                                    Stage.TEST,
                                                    self.DATASET_BUILDERS_VALIDATION_AND_TEST,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, context)


class ParameterizedPositiveNegativeTemplateDataSourcesFactory(BaseTemplateDataSourcesFactory):
    TRAIN_DATASET_BUILDERS = [ConditionalSequenceOrSequencePositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder()]
    TEST_VALID_DATASET_BUILDERS = [NextPositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder()]

    def _build_train_datasource(self, config: Config, context: Context) -> DataLoader:
        par_pos_neg_sampler_processor = {
            'type': "par_pos_neg",
            'seed': config.get('seed'),
            't': config.get('t')
        }

        loader_config = build_default_loader_config(config,
                                                    Stage.TRAIN,
                                                    self.TRAIN_DATASET_BUILDERS,
                                                    [par_pos_neg_sampler_processor])

        return self._build_datasource(loader_config, context)

    def _build_validation_datasource(self, config: Config, context: Context) -> DataLoader:
        loader_config = build_default_loader_config(config,
                                                    Stage.VALIDATION,
                                                    self.TEST_VALID_DATASET_BUILDERS,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, context)

    def _build_test_datasource(self, config: Config, context: Context) -> DataLoader:
        loader_config = build_default_loader_config(config,
                                                    Stage.TEST,
                                                    self.TEST_VALID_DATASET_BUILDERS,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, context)


class SlidingWindowTemplateDataSourcesFactory(BaseTemplateDataSourcesFactory):
    TARGET_INTERACTION_CONFIG_KEY = 'number_target_interactions'
    WINDOW_CONFIG_KEY = 'window_size'

    TRAIN_DATASET_BUILDERS = [SequenceDatasetRatioSplitBuilder(), LeaveOneOutSessionDatasetBuilder()]
    TEST_VALID_DATASET_BUILDERS = [NextPositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder()]

    def _build_train_datasource(self, config: Config, context: Context) -> DataLoader:
        window_size = config.get(self.WINDOW_CONFIG_KEY)
        number_target_interactions = config.get_or_default(self.TARGET_INTERACTION_CONFIG_KEY, 1)
        sequence_length = window_size + number_target_interactions
        config.set(self.WINDOW_CONFIG_KEY, sequence_length)

        fixed_sequence_length_processor = _build_fixed_sequence_length_processor_config(sequence_length)

        par_pos_neg_sampler_processor = {
            'type': "par_pos_neg",
            't': number_target_interactions
        }

        builders = [NextPositionWindowDatasetBuilder(),
                    LeaveOneOutSequenceWindowDatasetBuilder()]

        processors = [fixed_sequence_length_processor, par_pos_neg_sampler_processor]

        loader_config = build_default_loader_config(config, Stage.TRAIN, builders, processors)
        return self._build_datasource(loader_config, context)

    def _build_validation_datasource(self, config: Config, context: Context) -> DataLoader:
        window_size = config.get(self.WINDOW_CONFIG_KEY)
        number_target_interactions = config.get_or_default(self.TARGET_INTERACTION_CONFIG_KEY, 1)
        sequence_length = window_size + number_target_interactions

        fixed_sequence_length_processor = _build_fixed_sequence_length_processor_config(sequence_length)
        loader_config = build_default_loader_config(config,
                                                    Stage.VALIDATION,
                                                    self.TEST_VALID_DATASET_BUILDERS,
                                                    [fixed_sequence_length_processor, TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, context)

    def _build_test_datasource(self, config: Config, context: Context) -> DataLoader:
        window_size = config.get(self.WINDOW_CONFIG_KEY)
        number_target_interactions = config.get_or_default(self.TARGET_INTERACTION_CONFIG_KEY, 1)
        sequence_length = window_size + number_target_interactions

        fixed_sequence_length_processor = _build_fixed_sequence_length_processor_config(sequence_length)
        loader_config = build_default_loader_config(config,
                                                    Stage.TEST,
                                                    self.TEST_VALID_DATASET_BUILDERS,
                                                    [fixed_sequence_length_processor, TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, context)


def _build_fixed_sequence_length_processor_config(sequence_length: int) -> Dict[str, Any]:
    return {
        'type': FIXED_SEQUENCE_LENGTH_PROCESSOR_KEY,
        'fixed_length': sequence_length
    }
