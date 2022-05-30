from abc import abstractmethod
from typing import Any, List, Union, Dict

from torch.utils.data import DataLoader

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.factories.common.conditional_based_factory import ConditionalFactory
from asme.core.init.factories.data_sources.common import build_default_loader_config
from asme.core.init.factories.data_sources.datasets.processor.processors import FIXED_SEQUENCE_LENGTH_PROCESSOR_KEY
from asme.core.init.factories.data_sources.loader import LoaderFactory
from asme.core.init.factories.data_sources.registry import REGISTERED_TEMPLATES
from asme.core.init.factories.util import can_build_with_subsection, build_with_subsection
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.core.init.templating.datasources.datasources import Stage, SequenceDatasetRatioSplitBuilder, \
    LeaveOneOutSessionDatasetBuilder, \
    NextPositionDatasetBuilder, TARGET_EXTRACTOR_PROCESSOR_CONFIG, LeaveOneOutNextPositionDatasetBuilder, \
    ConditionalSequenceOrSequencePositionDatasetBuilder, POS_NEG_PROCESSOR_CONFIG, NextPositionWindowDatasetBuilder, \
    LeaveOneOutSequenceWindowDatasetBuilder, LeavePercentageOutSessionDatasetBuilder, \
    LeavePercentageOutNextPositionDatasetBuilder
from asme.data import CURRENT_SPLIT_PATH_CONTEXT_KEY


class TemplateDataSourcesFactory(ObjectFactory):

    def __init__(self, key: str):
        super().__init__()
        self._key = key
        self._factory = ConditionalFactory(key, REGISTERED_TEMPLATES)

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        return can_build_with_subsection(self._factory, build_context)

    def build(self, build_context: BuildContext) -> Union[Any, Dict[str, Any], List[Any]]:
        config = build_context.get_current_config_section()
        context = build_context.get_context()
        # If no dataset path was specified, try to the use the one provided by the datamodule
        config.set_if_absent("path", context.get(CURRENT_SPLIT_PATH_CONTEXT_KEY))

        return build_with_subsection(self._factory, build_context)

    def is_required(self, build_context: BuildContext) -> bool:
        return True

    def config_path(self) -> List[str]:
        return ["template"]

    def config_key(self) -> str:
        return "template"


class BaseTemplateDataSourcesFactory(ObjectFactory):

    def __init__(self):
        super().__init__()
        self._loader_factory = LoaderFactory()

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, build_context: BuildContext) -> Dict[str, DataLoader]:
        return {
            "train": self._build_train_datasource(build_context),
            "validation": self._build_validation_datasource(build_context),
            "test": self._build_test_datasource(build_context)
        }

    def is_required(self, build_context: BuildContext) -> bool:
        return True

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return ""

    # TODO we need to find a way to patch the build_context config so that we can change the loader config dynamically!
    def _build_datasource(self, config: Config, context: Context) -> DataLoader:
        loader_build_context = BuildContext(Config({"loader": config.get([])}), context)
        return build_with_subsection(self._loader_factory, loader_build_context)

    @abstractmethod
    def _build_train_datasource(self, build_context: BuildContext) -> DataLoader:
        pass

    @abstractmethod
    def _build_validation_datasource(self, build_context: BuildContext) -> DataLoader:
        pass

    @abstractmethod
    def _build_test_datasource(self, build_context: BuildContext) -> DataLoader:
        pass


class MaskTemplateDataSourcesFactory(BaseTemplateDataSourcesFactory):
    TRAIN_DATASET_BUILDERS = [SequenceDatasetRatioSplitBuilder(), LeaveOneOutSessionDatasetBuilder(),
                              LeavePercentageOutSessionDatasetBuilder()]
    TEST_VALID_DATASET_BUILDERS = [NextPositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder(),
                                   LeavePercentageOutSessionDatasetBuilder()]

    def __init__(self):
        super().__init__()

    def _build_train_datasource(self, build_context: BuildContext) -> DataLoader:
        config = build_context.get_current_config_section()
        mask_probability = config.get_or_default('mask_probability', 0.2)
        only_last_item_mask_prob = config.get_or_default('only_last_item_mask_prob', 0.1)
        cloze_processor = {
            'type': "cloze",
            'mask_probability': mask_probability,
            'only_last_item_mask_prob': only_last_item_mask_prob,
        }
        loader_config = build_default_loader_config(config, Stage.TRAIN, self.TRAIN_DATASET_BUILDERS, [cloze_processor])

        return self._build_datasource(loader_config, build_context.get_context())

    def _build_validation_datasource(self, build_context: BuildContext) -> DataLoader:
        config = build_context.get_current_config_section()
        mask_last_item_processor = {
            'type': 'last_item_mask'
        }
        loader_config = build_default_loader_config(config, Stage.VALIDATION, self.TEST_VALID_DATASET_BUILDERS,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG, mask_last_item_processor])

        return self._build_datasource(loader_config, build_context.get_context())

    def _build_test_datasource(self, build_context: BuildContext) -> DataLoader:
        config = build_context.get_current_config_section()

        mask_last_item_processor = {
            'type': 'last_item_mask'
        }
        loader_config = build_default_loader_config(config, Stage.TEST, self.TEST_VALID_DATASET_BUILDERS,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG, mask_last_item_processor])

        return self._build_datasource(loader_config, build_context.get_context())


class NextSequenceStepTemplateDataSourcesFactory(BaseTemplateDataSourcesFactory):
    TRAIN_DATASET_BUILDERS = [NextPositionDatasetBuilder(), LeaveOneOutNextPositionDatasetBuilder(),
                              LeavePercentageOutNextPositionDatasetBuilder()]
    TEST_VALID_DATASET_BUILDERS = [NextPositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder(),
                                   LeavePercentageOutSessionDatasetBuilder()]

    def __init__(self):
        super().__init__()

    def _build_train_datasource(self, build_context: BuildContext) -> DataLoader:
        loader_config = build_default_loader_config(build_context.get_current_config_section(), Stage.TRAIN, self.TRAIN_DATASET_BUILDERS,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, build_context.get_context())

    def _build_validation_datasource(self, build_context: BuildContext) -> DataLoader:
        loader_config = build_default_loader_config(build_context.get_current_config_section(), Stage.VALIDATION, self.TEST_VALID_DATASET_BUILDERS,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, build_context.get_context())

    def _build_test_datasource(self, build_context: BuildContext) -> DataLoader:
        loader_config = build_default_loader_config(build_context.get_current_config_section(), Stage.TEST, self.TEST_VALID_DATASET_BUILDERS,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, build_context.get_context())


class PositiveNegativeTemplateDataSourcesFactory(BaseTemplateDataSourcesFactory):
    TRAIN_DATASET_BUILDERS = [ConditionalSequenceOrSequencePositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder(),
                              LeavePercentageOutSessionDatasetBuilder()]
    TEST_VALID_DATASET_BUILDERS = [NextPositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder(),
                                   LeavePercentageOutSessionDatasetBuilder()]

    def __init__(self):
        super().__init__()

    def _build_train_datasource(self, build_context: BuildContext) -> DataLoader:
        loader_config = build_default_loader_config(build_context.get_current_config_section(),
                                                    Stage.TRAIN,
                                                    self.TRAIN_DATASET_BUILDERS,
                                                    [POS_NEG_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, build_context.get_context())

    def _build_validation_datasource(self, build_context: BuildContext) -> DataLoader:
        loader_config = build_default_loader_config(build_context.get_current_config_section(),
                                                    Stage.VALIDATION,
                                                    self.TEST_VALID_DATASET_BUILDERS,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, build_context.get_context())

    def _build_test_datasource(self, build_context: BuildContext) -> DataLoader:
        loader_config = build_default_loader_config(build_context.get_current_config_section(),
                                                    Stage.TEST,
                                                    self.TEST_VALID_DATASET_BUILDERS,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, build_context.get_context())


class ParallelSeqTrainingTemplateDataSourcesFactory(BaseTemplateDataSourcesFactory):
    # (AD) TODO configure different dataset builders
    DATASET_BUILDER_TRAINING = [SequenceDatasetRatioSplitBuilder()]
    DATASET_BUILDERS_VALIDATION_AND_TEST = [NextPositionDatasetBuilder()]

    def _build_train_datasource(self, build_context: BuildContext) -> DataLoader:
        target_extractor_processor = {
            "type": "target_extractor",
            "parallel": True
        }
        loader_config = build_default_loader_config(build_context.get_current_config_section(), Stage.TRAIN, self.DATASET_BUILDER_TRAINING, [target_extractor_processor])
        return self._build_datasource(loader_config, build_context.get_context())

    def _build_validation_datasource(self, build_context: BuildContext) -> DataLoader:
        loader_config = build_default_loader_config(build_context.get_current_config_section(),
                                                    Stage.VALIDATION,
                                                    self.DATASET_BUILDERS_VALIDATION_AND_TEST,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, build_context.get_context())

    def _build_test_datasource(self, build_context: BuildContext) -> DataLoader:
        loader_config = build_default_loader_config(build_context.get_current_config_section(),
                                                    Stage.TEST,
                                                    self.DATASET_BUILDERS_VALIDATION_AND_TEST,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, build_context.get_context())


class PlainTrainingTemplateDataSourcesFactory(BaseTemplateDataSourcesFactory):
    DATASET_BUILDER_TRAINING = [SequenceDatasetRatioSplitBuilder()]
    DATASET_BUILDERS_VALIDATION_AND_TEST = [NextPositionDatasetBuilder()]

    def _build_train_datasource(self, build_context: BuildContext) -> DataLoader:
        loader_config = build_default_loader_config(build_context.get_current_config_section(), Stage.TRAIN, self.DATASET_BUILDER_TRAINING)
        return self._build_datasource(loader_config, build_context.get_context())

    def _build_validation_datasource(self, build_context: BuildContext) -> DataLoader:
        loader_config = build_default_loader_config(build_context.get_current_config_section(),
                                                    Stage.VALIDATION,
                                                    self.DATASET_BUILDERS_VALIDATION_AND_TEST,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, build_context.get_context())

    def _build_test_datasource(self, build_context: BuildContext) -> DataLoader:
        loader_config = build_default_loader_config(build_context.get_current_config_section(),
                                                    Stage.TEST,
                                                    self.DATASET_BUILDERS_VALIDATION_AND_TEST,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, build_context.get_context())


class ParameterizedPositiveNegativeTemplateDataSourcesFactory(BaseTemplateDataSourcesFactory):
    TRAIN_DATASET_BUILDERS = [ConditionalSequenceOrSequencePositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder(),
                              LeavePercentageOutSessionDatasetBuilder()]
    TEST_VALID_DATASET_BUILDERS = [NextPositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder(),
                                   LeavePercentageOutSessionDatasetBuilder()]

    def _build_train_datasource(self, build_context: BuildContext) -> DataLoader:
        config = build_context.get_current_config_section()
        neg_sampler_processor = {
            'type': "negative_item_sampler",
            'number_negative_items': config.get_or_default('number_negative_items', 1)
        }

        pos_extractor_processor = {
            'type': 'positive_item_extractor',
            'number_positive_items': config.get_or_default('number_positive_items', 1)
        }

        loader_config = build_default_loader_config(config,
                                                    Stage.TRAIN,
                                                    self.TRAIN_DATASET_BUILDERS,
                                                    [pos_extractor_processor, neg_sampler_processor])

        return self._build_datasource(loader_config, build_context.get_context())

    def _build_validation_datasource(self, build_context: BuildContext) -> DataLoader:
        loader_config = build_default_loader_config(build_context.get_current_config_section(),
                                                    Stage.VALIDATION,
                                                    self.TEST_VALID_DATASET_BUILDERS,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, build_context.get_context())

    def _build_test_datasource(self, build_context: BuildContext) -> DataLoader:
        loader_config = build_default_loader_config(build_context.get_current_config_section(),
                                                    Stage.TEST,
                                                    self.TEST_VALID_DATASET_BUILDERS,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, build_context.get_context())


class SlidingWindowTemplateDataSourcesFactory(BaseTemplateDataSourcesFactory):
    """

    """
    WINDOW_TARGET_LENGTH_CONFIG_KEY = 'window_target_length'
    WINDOW_MARKOV_LENGTH_CONFIG_KEY = 'window_markov_length'

    TRAIN_DATASET_BUILDERS = [SequenceDatasetRatioSplitBuilder(), LeaveOneOutSessionDatasetBuilder(),
                              LeavePercentageOutSessionDatasetBuilder()]
    TEST_VALID_DATASET_BUILDERS = [NextPositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder(),
                                   LeavePercentageOutSessionDatasetBuilder()]

    def _build_train_datasource(self, build_context: BuildContext) -> DataLoader:
        config = build_context.get_current_config_section()
        window_markov_length = config.get_or_default(self.WINDOW_MARKOV_LENGTH_CONFIG_KEY, 2)
        window_target_length = config.get_or_default(self.WINDOW_TARGET_LENGTH_CONFIG_KEY, 1)
        window_size = window_markov_length + window_target_length
        fixed_sequence_length_processor = _build_fixed_sequence_length_processor_config(window_size)

        pos_extractor_processor = {
            'type': 'positive_item_extractor',
            'number_positive_items': config.get_or_default('number_positive_items', 1)
        }

        neg_sampler_processor = {
            'type': "negative_item_sampler",
            'number_negative_items': config.get_or_default('number_negative_items', 1)
        }

        builders = [NextPositionWindowDatasetBuilder(),
                    LeaveOneOutSequenceWindowDatasetBuilder()]

        processors = [fixed_sequence_length_processor, pos_extractor_processor, neg_sampler_processor]

        loader_config = build_default_loader_config(config, Stage.TRAIN, builders, processors)
        return self._build_datasource(loader_config, build_context.get_context())

    def _build_validation_datasource(self, build_context: BuildContext) -> DataLoader:
        config = build_context.get_current_config_section()

        window_markov_length = config.get_or_default(self.WINDOW_MARKOV_LENGTH_CONFIG_KEY, 2)
        window_target_length = config.get_or_default(self.WINDOW_TARGET_LENGTH_CONFIG_KEY, 1)
        window_size = window_markov_length + window_target_length
        fixed_sequence_length_processor = _build_fixed_sequence_length_processor_config(window_size)
        loader_config = build_default_loader_config(config,
                                                    Stage.VALIDATION,
                                                    self.TEST_VALID_DATASET_BUILDERS,
                                                    [fixed_sequence_length_processor,
                                                     TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, build_context.get_context())

    def _build_test_datasource(self, build_context: BuildContext) -> DataLoader:
        config = build_context.get_current_config_section()

        window_markov_length = config.get_or_default(self.WINDOW_MARKOV_LENGTH_CONFIG_KEY, 2)
        window_target_length = config.get_or_default(self.WINDOW_TARGET_LENGTH_CONFIG_KEY, 1)
        window_size = window_markov_length + window_target_length
        fixed_sequence_length_processor = _build_fixed_sequence_length_processor_config(window_size)
        loader_config = build_default_loader_config(config,
                                                    Stage.TEST,
                                                    self.TEST_VALID_DATASET_BUILDERS,
                                                    [fixed_sequence_length_processor,
                                                     TARGET_EXTRACTOR_PROCESSOR_CONFIG])
        return self._build_datasource(loader_config, build_context.get_context())


def _build_fixed_sequence_length_processor_config(sequence_length: int) -> Dict[str, Any]:
    return {
        'type': FIXED_SEQUENCE_LENGTH_PROCESSOR_KEY,
        'fixed_length': sequence_length
    }
