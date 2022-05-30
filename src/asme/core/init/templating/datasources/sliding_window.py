from typing import Dict, Any

from asme.core.init.factories.data_sources.datasets.processor.processors import FIXED_SEQUENCE_LENGTH_PROCESSOR_KEY

from asme.core.init.templating.datasources.datasources import build_datasource, DataSourceTemplateProcessor, \
    SequenceDatasetRatioSplitBuilder, LeaveOneOutSessionDatasetBuilder, NextPositionDatasetBuilder, \
    TARGET_EXTRACTOR_PROCESSOR_CONFIG, LeaveOneOutSequenceWindowDatasetBuilder, NextPositionWindowDatasetBuilder, Stage


def _build_fixed_sequence_length_processor_config(sequence_length: int) -> Dict[str, Any]:
    return {
        'type': FIXED_SEQUENCE_LENGTH_PROCESSOR_KEY,
        'fixed_length': sequence_length
    }


class SlidingWindowDataSourceTemplateProcessor(DataSourceTemplateProcessor):

    """
    TODO: write documentation
    """

    WINDOW_TARGET_LENGTH_CONFIG_KEY = 'window_target_length'
    WINDOW_MARKOV_LENGTH_CONFIG_KEY = 'window_markov_length'

    TRAIN_DATASET_BUILDERS = [NextPositionWindowDatasetBuilder(), LeaveOneOutSequenceWindowDatasetBuilder()]
    TEST_VALID_DATASET_BUILDERS = [NextPositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder()]

    def _get_template_key(self) -> str:
        return 'sliding_window_data_sources'

    def _build_train_datasource(self, config: Dict[str, Any]) -> Dict[str, Any]:
        window_markov_length = config[self.WINDOW_MARKOV_LENGTH_CONFIG_KEY]
        window_target_length = config[self.WINDOW_TARGET_LENGTH_CONFIG_KEY]
        window_size = window_markov_length + window_target_length
        fixed_sequence_length_processor = _build_fixed_sequence_length_processor_config(window_size)

        pos_extractor_processor = {
            'type': 'positive_item_extractor',
            'number_positive_items': config['number_positive_items']
        }

        neg_sampler_processor = {
            'type': "negative_item_sampler",
            'number_negative_items': config['number_negative_items']
        }

        processors = [fixed_sequence_length_processor, pos_extractor_processor, neg_sampler_processor]

        return build_datasource(self.TRAIN_DATASET_BUILDERS, config, Stage.TRAIN, processors)

    def _build_validation_datasource(self, config: Dict[str, Any]) -> Dict[str, Any]:
        window_markov_length = config[self.WINDOW_MARKOV_LENGTH_CONFIG_KEY]
        window_target_length = config[self.WINDOW_TARGET_LENGTH_CONFIG_KEY]
        window_size = window_markov_length + window_target_length

        fixed_sequence_length_processor = _build_fixed_sequence_length_processor_config(window_size)
        return build_datasource(self.TEST_VALID_DATASET_BUILDERS, config, Stage.VALIDATION,
                                [fixed_sequence_length_processor, TARGET_EXTRACTOR_PROCESSOR_CONFIG])

    def _build_test_datasource(self, config: Dict[str, Any]) -> Dict[str, Any]:
        window_markov_length = config[self.WINDOW_MARKOV_LENGTH_CONFIG_KEY]
        window_target_length = config[self.WINDOW_TARGET_LENGTH_CONFIG_KEY]
        window_size = window_markov_length + window_target_length
        fixed_sequence_length_processor = _build_fixed_sequence_length_processor_config(window_size)
        return build_datasource(self.TEST_VALID_DATASET_BUILDERS, config, Stage.TEST,
                                [fixed_sequence_length_processor, TARGET_EXTRACTOR_PROCESSOR_CONFIG])
