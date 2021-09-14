from typing import Dict, Any

from asme.data.datasets import TARGET_SUFFIX
from asme.data.datasets.processors.processor import Processor


class TargetExtractorProcessor(Processor):

    """
    This processor extracts the last item in the sequence as target for the training or evaluation process

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
                sub_sequence = value[:last_pos]
                target = value[last_pos]

                processed_information[key] = sub_sequence
                processed_information[key + TARGET_SUFFIX] = target
            else:
                processed_information[key] = value

        return processed_information
