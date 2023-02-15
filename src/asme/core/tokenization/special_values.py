from asme.data.utils.converter_utils import build_converter

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




