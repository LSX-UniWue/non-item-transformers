from typing import Dict, Any

from asme.data.datasets import TARGET_SUFFIX
from asme.data.datasets.processors.processor import Processor


class NoTargetExtractorProcessor(Processor):

    """
    This processor use the last item in the sequence as a dummy target, if you do not have a target item, but want to
    use the predict function of the framework

    e.g. if the session is [5, 4, 3, 7] the processor sets the list to [5, 4, 3] and adds a TARGET_ENTRY_NAME to 7
    """

    def process(self,
                parsed_sequence: Dict[str, Any]
                ) -> Dict[str, Any]:

        processed_information = {}

        for key, value in parsed_sequence.items():
            if isinstance(value, list):
                sequence_length = len(value)
                last_pos = sequence_length - 1
                processed_information[key] = value
                # Add last token to get some valid token
                processed_information[key + TARGET_SUFFIX] = value[last_pos]
            else:
                processed_information[key] = value

        return processed_information
