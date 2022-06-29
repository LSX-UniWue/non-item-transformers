import ast, json
from typing import Dict, Any, Optional
import pandas as pd
import csv

from asme.data.utils.converter_utils import build_converter


class SpecialValues:

    def __init__(self, pad_value, mask_value, none_value):
        self.pad_value = pad_value
        self.mask_value = mask_value
        self.none_value = none_value


class VectorDictionary:
    """
    TODO: add docu
    """

    def __init__(self,
                 dict_path: str,
                 type: str,
                 delimiter: str,
                 element_type: str,
                 pad_value: str,
                 mask_value: str,
                 unk_value: str,
                 ):
        super().__init__()
        self.path = dict_path
        self.delimiter = delimiter
        self.type = type
        self.element_type = element_type

        configs = {"element_type": self.element_type}
        converter = build_converter(self.type, configs)

        self.feature_dict: Dict[Any] = {}
        with open(dict_path) as f:
            csvreader = csv.reader(f, delimiter=delimiter)
            for line in csvreader:
                key = line[0]
                self.feature_dict[key] = converter(line[1])

        values = list(self.feature_dict.values())
        average = list(map(lambda x: sum(x)/len(x), zip(*values)))

        if pad_value == "average":
            self.pad_value = average
        else:
            self.pad_value = converter(pad_value)

        if mask_value == "average":
            self.mask_value = average
        else:
            self.mask_value = converter(mask_value)

        if unk_value == "average":
            self.unk_value = average
        else:
            self.unk_value = converter(unk_value)


    def map_key_to_value(self, key):
        return self.feature_dict.get(key, self.unk_value)
