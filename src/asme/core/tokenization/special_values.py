from random import random
from typing import Any

from asme.data.utils.converter_utils import build_converter


def build_random(feature_type: str
                    ) -> Any:
    if feature_type == 'int':
        return int

    if feature_type == 'float':
        return float

    if feature_type == 'bool':
        return bool(random.randrange(0, 1))

    if feature_type == 'timestamp':
        return functools.partial(_parse_timestamp, date_format=configs.get('format'))

    if feature_type == 'list':
        element_type = configs.get("element_type")
        delimiter = configs.get('delimiter')
        converter = build_converter(element_type, configs)
        return functools.partial(_parse_list, delimiter=delimiter, converter=converter)

    if feature_type == 'pd.array':
        element_type = configs.get("element_type")
        converter = build_converter(element_type, configs)
        return functools.partial(_parse_pd_array, converter=converter)

    raise KeyError(f'{feature_type} not supported. Currently only bool, timestamp and int are supported. '
                   f'See documentation for more details')

class SpecialValues:
    """
    Special values for padding, masking etc.
    """

    def __init__(self,
                 type: str,
                 pad_value: str = None,
                 mask_value: str = None,
                 unk_value: str = None,
                 element_type: str = None,
                 ):

        self.type = type
        self.element_type = element_type

        configs = {"element_type": self.element_type}
        converter = build_converter(self.type, configs)

        if pad_value != None:
            self.pad_value = converter(pad_value)
        if pad_value != None:
            self.mask_value = converter(mask_value)
        if pad_value != None:
            self.unk_value = converter(unk_value)

    def get_pad_value(self):
        return self.pad_value

    def get_mask_value(self):
        return self.mask_value

    def get_unk_value(self):
        return self.unk_value

    def get_random_value(self):
        return




