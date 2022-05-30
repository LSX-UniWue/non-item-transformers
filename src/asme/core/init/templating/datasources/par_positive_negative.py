from typing import Dict, Any

from asme.core.init.templating.datasources.datasources import build_datasource, DataSourceTemplateProcessor, \
    LeaveOneOutSessionDatasetBuilder, NextPositionDatasetBuilder, ConditionalSequenceOrSequencePositionDatasetBuilder, \
    TARGET_EXTRACTOR_PROCESSOR_CONFIG, Stage


class ParameterizedPositiveNegativeDataSourcesTemplateProcessor(DataSourceTemplateProcessor):

    """
     This data sources template processor configs the datasets in the following was:
    - train: a session datasource with a tokenizer, a positive extractor and a negative sampler
    - validation: a nextitem datasource with a tokenizer processor and a target extractor
    - test: a nextitem datasource with a tokenizer processor and a target extractor
    """

    TRAIN_DATASET_BUILDERS = [ConditionalSequenceOrSequencePositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder()]
    TEST_VALID_DATASET_BUILDERS = [NextPositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder()]

    def _get_template_key(self) -> str:
        return 'par_pos_neg_data_sources'

    def _build_train_datasource(self, config: Dict[str, Any]) -> Dict[str, Any]:
        neg_sampler_processor = {
            'type': "negative_item_sampler",
            'number_negative_items': config['number_negative_items']
        }

        pos_extractor_processor = {
            'type': 'positive_item_extractor',
            'number_positive_items': config['number_positive_items']
        }

        return build_datasource(self.TRAIN_DATASET_BUILDERS, config, Stage.TRAIN, [neg_sampler_processor, pos_extractor_processor])

    def _build_validation_datasource(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return build_datasource(self.TEST_VALID_DATASET_BUILDERS, config, Stage.VALIDATION,
                                [TARGET_EXTRACTOR_PROCESSOR_CONFIG])

    def _build_test_datasource(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return build_datasource(self.TEST_VALID_DATASET_BUILDERS, config, Stage.TEST,
                                [TARGET_EXTRACTOR_PROCESSOR_CONFIG])
