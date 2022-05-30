from typing import Optional

from asme.core.models.common.layers.data.sequence import InputSequence, EmbeddedElementsSequence
from asme.core.models.common.layers.layers import SequenceElementsRepresentationLayer
from asme.core.models.common.layers.sequence_embedding import SequenceElementsEmbeddingLayer


# TODO rename file to elements_embedding
from asme.core.utils.hyperparameter_utils import save_hyperparameters


class SequenceElementsEmbeddingComponent(SequenceElementsRepresentationLayer):
    """
    A component that projects every element in a sequence into an embedding space. The component also supports pooling
    over elements of the sequence.
    """
    @save_hyperparameters
    def __init__(self,
                 vocabulary_size: int,
                 embedding_size: int,
                 pooling_type: Optional[str] = None,
                 dropout: Optional[float] = None):
        """

        :param vocabulary_size: the size of the elements vocabulary.
        :param embedding_size: the embedding size.
        :param pooling_type: the type of pooling that will be performed. (max, sum or mean)
        """
        super().__init__()
        self.elements_embedding = SequenceElementsEmbeddingLayer(vocabulary_size, embedding_size, pooling_type, dropout)

    def forward(self, input_sequence: InputSequence) -> EmbeddedElementsSequence:
        embedded_sequence = self.elements_embedding(input_sequence.sequence)

        return EmbeddedElementsSequence(embedded_sequence, input_sequence)
