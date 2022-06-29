from typing import Dict, Any
import ast

from asme.core.tokenization.vector_dictionary import VectorDictionary
from asme.data.datasets.processors.processor import Processor


class VectorDictionaryProcessor(Processor):

    def __init__(self,
                 dictionary_map: Dict[str, VectorDictionary]
                 ):
        super().__init__()
        self.dictionary_map = dictionary_map


    def process(self,
                parsed_sequence: Dict[str, Any]
                ) -> Dict[str, Any]:

        for feature in parsed_sequence:
            if feature in self.dictionary_map.keys():
                feature_dict = self.dictionary_map.get(feature)
                parsed_sequence[feature] = [feature_dict.map_key_to_value(item) for item in parsed_sequence[feature]]
        return parsed_sequence
