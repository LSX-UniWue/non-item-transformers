import torch
from asme.core.models.common.components.representations.sequence_elements_embedding_user import \
    EmbeddedElementsSequenceWithUserEmbedding
from asme.core.models.common.layers.data.sequence import SequenceRepresentation
from asme.core.models.common.layers.layers import SequenceRepresentationLayer
from torch import nn


class NNRecSequenceRepresentationComponent(SequenceRepresentationLayer):

    def __init__(self,
                 embedding_size: int,
                 hidden_size: int):
        super().__init__()
        self.hidden_layer = nn.Linear(embedding_size, hidden_size)
        self.act1 = nn.Tanh()

    def forward(self,
                embedded_sequence: EmbeddedElementsSequenceWithUserEmbedding
                ) -> SequenceRepresentation:
        sequence = embedded_sequence.embedded_sequence

        batch_size = sequence.size()[0]
        embedded_items = sequence.view(batch_size, -1)
        embedded_user = embedded_sequence.embedded_user

        if embedded_user is not None:
            overall_representation = torch.cat([embedded_user, embedded_items])
        else:
            overall_representation = embedded_items
        return SequenceRepresentation(self.act1(self.hidden_layer(overall_representation)))
