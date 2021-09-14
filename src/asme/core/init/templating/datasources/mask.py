from typing import Dict, Any

from asme.core.init.templating.datasources.datasources import build_datasource, DataSourceTemplateProcessor, \
    SequenceDatasetRatioSplitBuilder, LeaveOneOutSessionDatasetBuilder, NextPositionDatasetBuilder, \
    TARGET_EXTRACTOR_PROCESSOR_CONFIG, Stage


class MaskDataSourcesTemplateProcessor(DataSourceTemplateProcessor):

    """
    This data sources template processor configs the datasets in the following was:
    - train: a nextitem datasource with a tokenizer and cloze processor
    - validation: a nextitem datasource with a tokenizer, a target extractor and a list item mask processor
    - test: a nextitem datasource with a tokenizer, a target extractor and a list item mask processor
    """

    TRAIN_DATASET_BUILDERS = [SequenceDatasetRatioSplitBuilder(), LeaveOneOutSessionDatasetBuilder()]
    TEST_VALID_DATASET_BUILDERS = [NextPositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder()]

    def _get_template_key(self) -> str:
        return 'mask_data_sources'

    def _build_train_datasource(self, config: Dict[str, Any]) -> Dict[str, Any]:
        mask_probability = config.get('mask_probability', 0.2)
        only_last_item_mask_prob = config.get('only_last_item_mask_prob', 0.1)
        cloze_processor = {
            'type': "cloze",
            'mask_probability': mask_probability,
            'only_last_item_mask_prob': only_last_item_mask_prob,
        }

        return build_datasource(self.TRAIN_DATASET_BUILDERS, config, Stage.TRAIN, [cloze_processor])

    def _build_validation_datasource(self, config: Dict[str, Any]) -> Dict[str, Any]:
        mask_last_item_processor = {
            'type': 'last_item_mask'
        }
        return build_datasource(self.TEST_VALID_DATASET_BUILDERS, config, Stage.VALIDATION,
                                [TARGET_EXTRACTOR_PROCESSOR_CONFIG, mask_last_item_processor])

    def _build_test_datasource(self, config: Dict[str, Any]) -> Dict[str, Any]:
        mask_last_item_processor = {
            'type': 'last_item_mask'
        }
        return build_datasource(self.TEST_VALID_DATASET_BUILDERS, config, Stage.TEST,
                                [TARGET_EXTRACTOR_PROCESSOR_CONFIG, mask_last_item_processor])
