from typing import List

from asme.core.init.factories import BuildContext
from asme.core.init.factories.data_sources.datasets.processor.last_item_mask import get_all_tokenizers_from_context, \
    get_sequence_feature_names, get_all_feature_special_values_from_context
from asme.data.datasets.processors.cloze_mask import ClozeMaskProcessor
from asme.core.init.factories.features.tokenizer_factory import get_tokenizer_key_for_voc, ITEM_TOKENIZER_ID
from asme.core.init.factories.util import check_config_keys_exist, get_all_tokenizers_from_context
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class ClozeProcessorFactory(ObjectFactory):
    """
    factory for the ClozeMaskProcessor
    """

    def can_build(self,
                  build_context: BuildContext
                  ) -> CanBuildResult:
        # check for config keys
        config_keys_exist = check_config_keys_exist(build_context.get_current_config_section(), ['mask_probability', 'only_last_item_mask_prob'])
        if not config_keys_exist:
            return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION)

        if not build_context.get_context().has_path(get_tokenizer_key_for_voc(ITEM_TOKENIZER_ID)):
            return CanBuildResult(CanBuildResultType.MISSING_DEPENDENCY, 'item tokenizer missing')

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              build_context: BuildContext
              ) -> ClozeMaskProcessor:

        config = build_context.get_current_config_section()
        context = build_context.get_context()
        tokenizers = get_all_tokenizers_from_context(context)
        special_values = get_all_feature_special_values_from_context(context)

        mask_probability = config.get('mask_probability')
        exclude_features = config.get_or_default('exclude_features',list())
        only_last_item_mask_prob = config.get('only_last_item_mask_prob')

        masking_targets = get_sequence_feature_names(config, context)
        masking_targets = [masking_target for masking_target in masking_targets if masking_target not in exclude_features]

        return ClozeMaskProcessor(tokenizers, special_values, mask_probability, only_last_item_mask_prob, masking_targets)

    def is_required(self, build_context: BuildContext) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'cloze_processor'
