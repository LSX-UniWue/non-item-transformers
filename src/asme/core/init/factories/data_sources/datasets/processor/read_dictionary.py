from typing import List

from asme.core.init.factories import BuildContext
from asme.data.datasets import ITEM_SEQ_ENTRY_NAME
from asme.data.datasets.processors.read_dictionary import VectorDictionaryProcessor
from asme.core.init.context import Context
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class ReadDictionaryProcessorFactory(ObjectFactory):

    """
    Factory for the TokenizerProcessor
    """

    def can_build(self,
                  build_context: BuildContext
                  ) -> CanBuildResult:

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              build_context: BuildContext
              ) -> VectorDictionaryProcessor:

        features = build_context.get_context().as_dict()["features"]

        dictionary_map = {}
        for feature in features:
            feature_dict = feature.dictionary
            if feature_dict:
                dictionary_map[feature.feature_name] = feature_dict
        return VectorDictionaryProcessor(dictionary_map)

    def is_required(self, build_context: BuildContext) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'read_dictionary_processor'
