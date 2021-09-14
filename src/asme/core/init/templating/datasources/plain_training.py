from typing import Dict, Any

from asme.core.init.templating.datasources.datasources import DataSourceTemplateProcessor, build_datasource, \
    NextPositionDatasetBuilder, SequenceDatasetRatioSplitBuilder, TARGET_EXTRACTOR_PROCESSOR_CONFIG, Stage


class PlainTrainingSourcesTemplateProcessor(DataSourceTemplateProcessor):

    """
     This data sources template processor configs the datasets in the following way:
    - train: a item sequence datasource with a tokenizer processor
    - validation: a nextitem datasource with a tokenizer processor and a target extract processor
    - test: a nextitem datasource with a tokenizer processor and a target extract processor
    """

    DATASET_BUILDER_TRAINING = [SequenceDatasetRatioSplitBuilder()]
    DATASET_BUILDERS_VALIDATION_AND_TEST = [NextPositionDatasetBuilder()]

    def _get_template_key(self) -> str:
        return 'plain_training_next_item_test_and_validation_data_sources'

    def _build_train_datasource(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return build_datasource(self.DATASET_BUILDER_TRAINING, config, Stage.TRAIN)

    def _build_validation_datasource(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return build_datasource(self.DATASET_BUILDERS_VALIDATION_AND_TEST, config, Stage.VALIDATION,
                                [TARGET_EXTRACTOR_PROCESSOR_CONFIG])

    def _build_test_datasource(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return build_datasource(self.DATASET_BUILDERS_VALIDATION_AND_TEST, config, Stage.TEST,
                                [TARGET_EXTRACTOR_PROCESSOR_CONFIG])
