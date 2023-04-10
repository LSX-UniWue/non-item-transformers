from typing import Dict, Any, List, Optional

from asme.core.utils import logging
from asme.data.datasets.sequence import MetaInformation

from asme.data.datasets import TARGET_SUFFIX
from asme.data.datasets.processors.processor import Processor

logger = logging.get_logger(__name__)


class ProductSequenceEndProcessor(Processor):

    def __init__(self, item_id_type: str, features: Optional[List[MetaInformation]] = None):
        super().__init__()
        self.features = features
        self.item_id_type = item_id_type

    def is_sequence(self, feature_name: str) -> bool:
        """
        Determines whether the feature is a sequence.

        :param feature_name: the name of the feature.
        :return: True iff the feature is a sequence, otherwise False.
        """
        for feature in self.features:
            if feature.feature_name == feature_name:
                return feature.is_sequence

        logger.warning(f"Unable to find meta-information for feature: {feature_name}. Assuming it is not a sequence.")
        return False

    def process(self,
                parsed_sequence: Dict[str, Any]
                ) -> Dict[str, Any]:

        processed_information = {}

        item_id_type = parsed_sequence[self.item_id_type]
        last_occurence = len(item_id_type) - item_id_type[::-1].index(1)

        for key, value in parsed_sequence.items():
            if isinstance(value, list) and self.is_sequence(key):
                processed_information[key] = value[:last_occurence]

        return processed_information
