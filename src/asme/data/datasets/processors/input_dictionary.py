from typing import Dict, Any

from asme.core.tokenization.item_dictionary import ItemDictionary
from asme.data.datasets.processors.processor import Processor


class InputDictionaryProcessor(Processor):

    def __init__(self,
                 dictionary_map: Dict[str, ItemDictionary]
                 ):
        super().__init__()
        self.item_dictionary = dictionary_map


    def process(self,
                parsed_sequence: Dict[str, Any]
                ) -> Dict[str, Any]:

        for feature in parsed_sequence:
            if feature in self.item_dictionary.keys():
                feature_dict = self.item_dictionary.get(feature)
                parsed_sequence[feature] = [feature_dict.map_key_to_value(item) for item in parsed_sequence[feature]]
        return parsed_sequence
