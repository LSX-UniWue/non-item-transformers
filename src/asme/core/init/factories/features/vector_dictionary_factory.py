from typing import List

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories import BuildContext

from asme.core.init.factories.util import require_config_keys, can_build_with_subsection, build_with_subsection
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.core.tokenization.vector_dictionary import VectorDictionary, SpecialValues



class VectorDictionaryFactory(ObjectFactory):
    """
    Builds a single tokenizer entry inside the tokenizers section.
    """
    KEY = "dictionary"


    REQUIRED_KEYS = ["dict_path","default_value"]

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        return require_config_keys(build_context.get_current_config_section(), self.REQUIRED_KEYS)

    def build(self, build_context: BuildContext) -> VectorDictionary:

        config = build_context.get_current_config_section()

        delimiter = config.get_or_default("delimiter", "\t")
        dict_path = config.get_or_raise("dict_path", f"<dict_path> could not be found in dictionary config section.")
        type = config.get_or_default("type","pd.array")
        element_type = config.get_or_raise("element_type",f"type could not be found in dictionary config section")
        pad_value = config.get_or_default("pad_value", "average")
        unk_value = config.get_or_default("unk_value", "average")
        mask_value = config.get_or_default("mask_value", "average")

        return VectorDictionary(dict_path=dict_path,delimiter=delimiter, type=type, pad_value=pad_value,
                                mask_value=mask_value,unk_value =unk_value, element_type=element_type)



    def is_required(self, build_context: BuildContext) -> bool:
        return False

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY

DICTIONARIES_PREFIX = 'dictionaries'


def get_dict_key_for_attribute(dict_id: str) -> str:
    return f'{DICTIONARIES_PREFIX}.{dict_id}'