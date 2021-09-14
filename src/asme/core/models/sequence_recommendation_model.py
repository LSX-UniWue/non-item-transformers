from abc import ABC
from typing import Union, Tuple, List

import torch
from torch import nn

from asme.core.models.common.layers.layers import SequenceElementsRepresentationLayer, SequenceRepresentationLayer, \
    SequenceRepresentationModifierLayer, ProjectionLayer
from asme.core.models.common.layers.data.sequence import InputSequence


class SequenceRecommenderModel(ABC, nn.Module):
    """
    Configurable sequence recommender model.
    """
    def __init__(self,
                 sequence_embedding_layer: SequenceElementsRepresentationLayer,
                 sequence_representation_layer: SequenceRepresentationLayer,
                 sequence_representation_modifier_layer: SequenceRepresentationModifierLayer,
                 projection_layer: ProjectionLayer):
        """

        :param sequence_embedding_layer: a layer that computes an embedding for the items in a sequence.`
        :param sequence_representation_layer:  a layer that computes a representation for a sequence.
        :param sequence_representation_modifier_layer: an optional layer that transforms the sequence representation.
        :param projection_layer: a layer that computes scores for items based on the sequence representation.
        """
        super().__init__()

        self._sequence_embedding_layer = sequence_embedding_layer
        self._sequence_representation_layer = sequence_representation_layer
        self._sequence_representation_modifier_layer = sequence_representation_modifier_layer
        self._projection_layer = projection_layer

    def forward(self, sequence: InputSequence) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """

        :param sequence: a sequence.
        :return:
        """
        # 1. embed the information
        embedded_sequence = self._sequence_embedding_layer(sequence)
        embedded_sequence.input_sequence = sequence

        # 2. get the sequence representation
        sequence_representation = self._sequence_representation_layer(embedded_sequence)
        sequence_representation.embedded_elements_sequence = embedded_sequence

        # 3. maybe modify the representation
        modified_sequence_representation = self._sequence_representation_modifier_layer(sequence_representation)
        modified_sequence_representation.sequence_representation = sequence_representation

        return self._projection_layer(modified_sequence_representation)

    def required_metadata_keys(self) -> List[str]:
        """
        Returns a list of keys that reference metadata in a batch that is required to apply the model.

        :return: a list with keys that reference required metadata in a batch. Default: []
        """
        return []

    def optional_metadata_keys(self) -> List[str]:
        """
        Returns a list of keys that reference metadata in a batch that is optional to apply the model.

        :return: a list with keys that reference optional metadata in a batch. Default: []
        """
        return []
