from typing import Dict, Any, List, Optional

from asme.core.utils import logging
from asme.data.datasets.sequence import MetaInformation

from asme.data.datasets import TARGET_SUFFIX
from asme.data.datasets.processors.processor import Processor

logger = logging.get_logger(__name__)


class TargetExtractorProcessor(Processor):

    """
    This processor extracts the last item in the sequence as target for the training or evaluation process.
    If `parallel` is set to `True` targets for every sequence position will be extracted.
    `keep_input` preserves the original input

    e.g. if the session is [5, 4, 3, 7] the processor sets the list to [5, 4, 3] and adds a TARGET_ENTRY_NAME to 7
    """
    def __init__(self, features: Optional[List[MetaInformation]] = None, parallel: bool = False, first_target: bool = False):
        super().__init__()
        self.features = features
        self.parallel = parallel
        self.first_target = first_target

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

        for key, value in parsed_sequence.items():
            if isinstance(value, list) and self.is_sequence(key):
                sequence_length = len(value)
                last_pos = sequence_length - 1
                sub_sequence = value[:last_pos]

                if self.parallel:
                    if self.first_target:
                        target = value
                        sub_sequence = [sub_sequence[0]] + sub_sequence
                    else:
                        target = value[1:]
                else:
                    if self.first_target:
                        sub_sequence = [sub_sequence[0]] + sub_sequence
                        target = value[last_pos]
                    else:
                        target = value[last_pos]
                processed_information[key] = sub_sequence
                processed_information[key + TARGET_SUFFIX] = target
            else:
                processed_information[key] = value

        return processed_information
