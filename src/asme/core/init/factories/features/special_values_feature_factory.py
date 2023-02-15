from typing import List

from asme.core.init.config import Config
from asme.core.init.factories import BuildContext
from asme.core.init.factories.util import require_config_keys
from asme.core.init.object_factory import ObjectFactory, CanBuildResult
from asme.core.tokenization.special_values import SpecialValues


class SpecialValuesFeatureFactory(ObjectFactory):
    """
    Builds a the special values for padding, masking etc. for non-tokenized features.
    """

    KEY = "special_values"
    SPECIAL_TOKENS_KEY = "special_values"
    REQUIRED_KEYS = [SPECIAL_TOKENS_KEY]

    def __init__(self):
        super().__init__()

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        return require_config_keys(build_context.get_current_config_section(), self.REQUIRED_KEYS)

    def build(self, build_context: BuildContext) -> SpecialValues:
        config = build_context.get_current_config_section()
        type = config.get_or_default("type", None)
        element_type = config.get_or_default("element_type", None)
        pad_value = config.get_or_default("pad_value", 0)
        unk_value = config.get_or_default("unk_value", 0)
        mask_value = config.get_or_default("mask_value", 0)

        return SpecialValues(type, pad_value, mask_value, unk_value, element_type)

    def is_required(self, build_context: BuildContext) -> bool:
        return False

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY

SPECIAL_VALUES_PREFIX = 'special_values'


def get_dict_key_for_attribute(special_value_id: str) -> str:
    return f'{SPECIAL_VALUES_PREFIX}.{special_value_id}'