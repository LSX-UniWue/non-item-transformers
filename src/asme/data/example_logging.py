from abc import abstractmethod
from typing import List, Dict, Any

from dataclasses import dataclass


@dataclass
class Example:

    sequence_data: Dict[str, Any]
    processed_data: Dict[str, Any]


class ExampleLogger:
    """
    marker interface that the dataset allows to extract examples generated from the dataset
    """

    @abstractmethod
    def get_data_examples(self,
                          num_examples: int = 1
                          ) -> List[Example]:
        pass
