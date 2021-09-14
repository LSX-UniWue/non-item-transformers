from dataclasses import dataclass
from typing import List, Any, Dict, Optional

import torch


@dataclass
class InputSequence:

    sequence: torch.Tensor
    """
    A sequence tensor with item ids. :math:`(N, S)` or :math:`(N, S, BS)`.
    """

    padding_mask: torch.Tensor
    """
    A mask that contains positions of padding tokens. :math:`(N, S)`.
    """

    attributes: Dict[str, Any]
    """
    Attributes that can be used to contextualize the sequence.
    """

    def get_attributes(self) -> List[str]:
        """
        Gets the a list with all attribute names.

        :return: a list with attribute names.
        """
        return list(self.attributes.keys())

    def has_attribute(self, name: str) -> bool:
        """
        Checks if an attribute is already present.
        :param name: the attribute name.
        :return: true if the attribute is present, false otherwise.
        """
        return name in self.attributes

    def get_attribute(self, name: str) -> Optional[Any]:
        """
        Gets the value for an attribute.

        :param name: the attribute name.
        :return: the attribute value.
        """
        if self.has_attribute(name):
            return self.attributes[name]

        return None

    def set_attribute(self, name: str, value: Any, overwrite: bool = False):
        """
        Sets an attribute.

        :param name: an attribute name.
        :param value: a value.
        :param overwrite: if the attribute should be overwritten if it is already present.

        :return:
        """
        if self.has_attribute(name) and not overwrite:
            raise Exception("Attribute is already set.")
        self.attributes[name] = value


@dataclass
class EmbeddedElementsSequence:

    embedded_sequence: torch.Tensor
    """
    The embedded sequence of the provided items (and maybe more) :math: `(N, S, H)`.
    """

    input_sequence: Optional[InputSequence] = None


#TODO (AD) I think the dimensions are wrong, discuss with Daniel
@dataclass
class SequenceRepresentation:  # -> Sequence Representation
    encoded_sequence: torch.Tensor
    """
    An encoded sequence representation. :math:`(N, S, R)`.
    """

    embedded_elements_sequence: Optional[EmbeddedElementsSequence] = None

    @property
    def input_sequence(self):
        if self.embedded_elements_sequence.input_sequence is None:
            return None

        return self.embedded_elements_sequence.input_sequence


@dataclass
class ModifiedSequenceRepresentation:

    modified_encoded_sequence: torch.Tensor
    """
    A modification of the encoded sequence. :math:`(N, S, T)`
    """

    sequence_representation: Optional[SequenceRepresentation] = None

    @property
    def input_sequence(self):
        if self.sequence_representation.input_sequence is None:
            return None

        return self.sequence_representation.input_sequence

