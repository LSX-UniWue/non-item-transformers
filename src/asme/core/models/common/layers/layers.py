import math
import abc
from typing import Union, Tuple
from abc import ABC

import torch
from torch import nn

from asme.core.models.common.layers.data.sequence import InputSequence, EmbeddedElementsSequence, SequenceRepresentation, \
    ModifiedSequenceRepresentation
from asme.core.utils.hyperparameter_utils import save_hyperparameters


class SequenceElementsRepresentationLayer(ABC, nn.Module):
    """
    Base class for modules that embed the elements of a sequence.
    """
    @abc.abstractmethod
    def forward(self, sequence: InputSequence) -> EmbeddedElementsSequence:
        """
        :param sequence: a sequence.

        :return: an embedded sequence.
        """
        pass


class SequenceRepresentationLayer(ABC, nn.Module):
    @abc.abstractmethod
    def forward(self, embedded_sequence: EmbeddedElementsSequence) -> SequenceRepresentation:
        """

        :param embedded_sequence: an embedded sequence..

        :return: a sequence representation. :math:`(N, S, R)`
        """
        pass


class SequenceRepresentationModifierLayer(ABC, nn.Module):
    @abc.abstractmethod
    def forward(self, sequence_representation: SequenceRepresentation) -> ModifiedSequenceRepresentation:
        """
        :param sequence_representation: an encoded sequence
        :return: a modified sequence representation. :math:`(N, S, T)`
        """
        pass


class ProjectionLayer(ABC, nn.Module):
    @abc.abstractmethod
    def forward(self,
                modified_sequence_representation: ModifiedSequenceRepresentation
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """

        :param modified_sequence_representation: a modified sequence representation tensor.

        :return: a score for each (provided) item.
        if no positive_samples and negative samples are provided a tensor of :math:`(N, I)` is returned
        if positive samples are provided :math:`(N, PI)` or
        if positive and negative samples are provided a tuple of two tensors of shape :math:`(N, PI)`, :math:`(N, NI)`
        """
        pass


class IdentitySequenceRepresentationModifierLayer(SequenceRepresentationModifierLayer):

    """ a SequenceRepresentationModifierLayer that does nothing with the sequence representation """

    def forward(self, sequence_representation: SequenceRepresentation) -> ModifiedSequenceRepresentation:
        return ModifiedSequenceRepresentation(sequence_representation.encoded_sequence, sequence_representation)


# TODO remove
class MatrixFactorizationLayer(nn.Linear, ABC):
    """
    a matrix factorization layer
    """
    @save_hyperparameters
    def __init__(self,
                 _weight: torch.Tensor,
                 bias: bool = False
                 ):
        super().__init__(_weight.size(0), _weight.size(1), bias=bias)
        self.weight = _weight


PROJECT_TYPE_LINEAR = 'linear'


class LinearProjectionLayer(ProjectionLayer):

    """
    A projection layer that uses a linear layer to project the sequence representation into a score for each item
    """
    @save_hyperparameters
    def __init__(self,
                 hidden_size: int,
                 item_vocab_size: int):
        super().__init__()

        self.linear = nn.Linear(hidden_size, item_vocab_size)

    def forward(self,
                modified_sequence_representation: ModifiedSequenceRepresentation
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        representation = modified_sequence_representation.modified_encoded_sequence
        return self.linear(representation)


class ItemEmbeddingProjectionLayer(ProjectionLayer):

    """
    A projection layer that uses the item embedding matrix to transform the sequence representation into the item space

    TODO: currently the sequence representation size must be the same as the item space size
    """

    @save_hyperparameters
    def __init__(self,
                 item_vocab_size: int,
                 embedding: nn.Embedding
                 ):
        super().__init__()

        self.item_vocab_size = item_vocab_size

        self.embedding = embedding
        self.output_bias = nn.Parameter(torch.Tensor(self.item_vocab_size))

        self.init_weights()

    def init_weights(self):
        bound = 1 / math.sqrt(self.item_vocab_size)
        nn.init.uniform_(self.output_bias, -bound, bound)

    def forward(self,
                modified_sequence_representation: ModifiedSequenceRepresentation
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        representation = modified_sequence_representation.modified_encoded_sequence
        dense = torch.matmul(representation, self.embedding.weight.transpose(0, 1))  # (S, N, I)
        return dense + self.output_bias


def build_projection_layer(project_type: str,
                           transformer_hidden_size: int,
                           item_voc_size: int,
                           embedding: nn.Embedding
                           ) -> ProjectionLayer:
    if project_type == PROJECT_TYPE_LINEAR:
        return LinearProjectionLayer(transformer_hidden_size, item_voc_size)

    if project_type == 'transpose_embedding':
        return ItemEmbeddingProjectionLayer(item_voc_size, embedding)

    raise KeyError(f'{project_type} invalid projection layer')
