

from typing import List

from asme.core.init.factories import BuildContext
from asme.core.init.factories.data_sources.datasets.processor.last_item_mask import get_all_tokenizers_from_context, \
    get_sequence_feature_names, get_all_feature_special_values_from_context
from asme.core.init.factories.features.features_factory import FeaturesFactory
from asme.data.datasets.processors.cloze_mask import ClozeMaskProcessor
from asme.core.init.factories.features.tokenizer_factory import get_tokenizer_key_for_voc, ITEM_TOKENIZER_ID
from asme.core.init.factories.util import check_config_keys_exist, get_all_tokenizers_from_context
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.data.datasets.processors.product_sequence_end import ProductSequenceEndProcessor


class ProductSequenceEndProcessorFactory(ObjectFactory):
    """
    factory for the product_sequence_end
    """

    def can_build(self,
                  build_context: BuildContext
                  ) -> CanBuildResult:
        # check for config keys
        config_keys_exist = check_config_keys_exist(build_context.get_current_config_section(), ['item_type_id'])
        if not config_keys_exist:
            return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION)

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              build_context: BuildContext
              ) -> ProductSequenceEndProcessor:

        config = build_context.get_current_config_section()
        context = build_context.get_context()

        features = build_context.get_context().get([FeaturesFactory.KEY])

        item_id_type = config.get('item_type_id')

        return ProductSequenceEndProcessor(item_id_type,features)

    def is_required(self, build_context: BuildContext) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'product_sequence_end'
