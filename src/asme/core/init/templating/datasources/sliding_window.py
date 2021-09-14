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

    TARGET_INTERACTION_CONFIG_KEY = 'number_target_interactions'
    WINDOW_CONFIG_KEY = 'window_size'

    TRAIN_DATASET_BUILDERS = [SequenceDatasetRatioSplitBuilder(), LeaveOneOutSessionDatasetBuilder()]
    TEST_VALID_DATASET_BUILDERS = [NextPositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder()]

    def _get_template_key(self) -> str:
        return 'sliding_window_data_sources'

    def _build_train_datasource(self, config: Dict[str, Any]) -> Dict[str, Any]:
        window_size = config[self.WINDOW_CONFIG_KEY]
        number_target_interactions = config.get(self.TARGET_INTERACTION_CONFIG_KEY, 1)
        sequence_length = window_size + number_target_interactions
        config[self.WINDOW_CONFIG_KEY] = window_size

        fixed_sequence_length_processor = _build_fixed_sequence_length_processor_config(sequence_length)

        par_pos_neg_sampler_processor = {
            'type': "par_pos_neg",
            't': number_target_interactions
        }

        builders = [NextPositionWindowDatasetBuilder(),
                    LeaveOneOutSequenceWindowDatasetBuilder()]

        processors = [fixed_sequence_length_processor, par_pos_neg_sampler_processor]

        return build_datasource(builders, config, Stage.TRAIN, processors)

    def _build_validation_datasource(self, config: Dict[str, Any]) -> Dict[str, Any]:
        window_size = config[self.WINDOW_CONFIG_KEY]
        number_target_interactions = config.get(self.TARGET_INTERACTION_CONFIG_KEY, 1)
        sequence_length = window_size + number_target_interactions

        fixed_sequence_length_processor = _build_fixed_sequence_length_processor_config(sequence_length)
        return build_datasource(self.TEST_VALID_DATASET_BUILDERS, config, Stage.VALIDATION,
                                [fixed_sequence_length_processor, TARGET_EXTRACTOR_PROCESSOR_CONFIG])

    def _build_test_datasource(self, config: Dict[str, Any]) -> Dict[str, Any]:
        window_size = config[self.WINDOW_CONFIG_KEY]
        number_target_interactions = config.get(self.TARGET_INTERACTION_CONFIG_KEY, 1)
        sequence_length = window_size + number_target_interactions

        fixed_sequence_length_processor = _build_fixed_sequence_length_processor_config(sequence_length)
        return build_datasource(self.TEST_VALID_DATASET_BUILDERS, config, Stage.TEST,
                                [fixed_sequence_length_processor, TARGET_EXTRACTOR_PROCESSOR_CONFIG])
