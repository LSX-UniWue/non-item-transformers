from typing import Optional

import torch
from asme.core.models.common.layers.data.sequence import InputSequence, EmbeddedElementsSequence
from asme.core.models.common.layers.layers import SequenceElementsRepresentationLayer
from asme.core.models.common.layers.sequence_embedding import SequenceElementsEmbeddingLayer
from asme.core.utils.hyperparameter_utils import save_hyperparameters
from asme.data.datasets import USER_ENTRY_NAME
from torch import nn as nn


class EmbeddedElementsSequenceWithUserEmbedding(EmbeddedElementsSequence):

    embedded_user: Optional[torch.Tensor] = None
    """
    the representation of an the user that interacted with the sequence :math:`(N, H)`
    """


class SequenceElementsEmbeddingWithUserEmbeddingComponent(SequenceElementsRepresentationLayer):

    """
    this component embeds the items in the sequence and the user if provided and returns it
    in an EmbeddedElementsSequenceWithUserEmbedding object
    """

    @save_hyperparameters
    def __init__(self,
                 user_vocab_size: int,
                 embedding_size: int,
                 item_embedding_layer: SequenceElementsEmbeddingLayer
                 ):
        super().__init__()

        self.item_embedding_layer = item_embedding_layer

        if embedding_size > 0:
            self.user_embeddings = nn.Embedding(user_vocab_size, embedding_size)

            # init weights
            self.user_embeddings.weight.data.normal_(0, 1.0 / embedding_size)

    def forward(self, sequence: InputSequence) -> EmbeddedElementsSequenceWithUserEmbedding:
        item_embedding = self.item_embedding_layer(sequence.sequence)

        sequence_result = EmbeddedElementsSequenceWithUserEmbedding(item_embedding)

        if sequence.has_attribute(USER_ENTRY_NAME):
            user = sequence.get_attribute(USER_ENTRY_NAME)
            user_embedding = self.user_embeddings(user)
            sequence_result.embedded_user = user_embedding

        return sequence_result
