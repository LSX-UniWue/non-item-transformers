from abc import abstractmethod
from typing import Dict, Any, List


class Processor:

    """
    Processors can be used to augment the raw input data. Examples include: masking tokens or augmenting the input data.

    The processors can be chained, but the Tokenizer Processor must be the first processor in the chain!

    """

    @abstractmethod
    def process(self,
                parsed_sequence: Dict[str, Any]
                ) -> Dict[str, Any]:
        """
        Takes the input and processes/enhances it.

        :param parsed_sequence: input data at this step of the pipeline.
        :return: a dictionary with the processed version of the input data.
        """
        pass


class DelegatingProcessor(Processor):

    def __init__(self,
                 processors: List[Processor]):
        super().__init__()

        self.processors = processors

    def process(self,
                parsed_sequence: Dict[str, Any]
                ) -> Dict[str, Any]:
        for processor in self.processors:
            parsed_sequence = processor.process(parsed_sequence)

        return parsed_sequence
