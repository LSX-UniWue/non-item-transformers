from typing import Dict, Any

from asme.core.models.common.layers.data.sequence import InputSequence, EmbeddedElementsSequence
from asme.core.models.common.layers.layers import SequenceElementsRepresentationLayer
from asme.core.models.common.layers.transformer_layers import TransformerEmbedding
from asme.core.models.kebert4rec.layers import LinearUpscaler
from torch import nn


def _build_embedding_type(embedding_type: str,
                          vocab_size: int,
                          hidden_size: int
                          ) -> nn.Module:
    return {
        'content_embedding': nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=hidden_size),
        'linear_upscale': LinearUpscaler(vocab_size=vocab_size,
                                         embed_size=hidden_size)
    }[embedding_type]


class KeBERT4RecSequenceElementsRepresentationComponent(SequenceElementsRepresentationLayer):

    def __init__(self,
                 item_embedding_layer: TransformerEmbedding,
                 embedding_size: int,
                 additional_attributes: Dict[str, Dict[str, Any]],
                 dropout: float = 0.0
                 ):
        super().__init__()

        self.item_embedding_layer = item_embedding_layer

        additional_attribute_embeddings = {}
        for attribute_name, attribute_infos in additional_attributes.items():
            embedding_type = attribute_infos['embedding_type']
            vocab_size = attribute_infos['vocab_size']
            additional_attribute_embeddings[attribute_name] = _build_embedding_type(embedding_type=embedding_type,
                                                                                    vocab_size=vocab_size,
                                                                                    hidden_size=embedding_size)
        self.additional_attribute_embeddings = nn.ModuleDict(additional_attribute_embeddings)

        self.dropout_embedding = nn.Dropout(dropout)
        self.norm_embedding = nn.LayerNorm(embedding_size)

    def forward(self, sequence: InputSequence) -> EmbeddedElementsSequence:
        embedding_sequence = self.item_embedding_layer(sequence)
        embedding = embedding_sequence.embedded_sequence
        for input_key, module in self.additional_attribute_embeddings.items():
            additional_metadata = sequence.get_attribute(input_key)
            embedding += module(additional_metadata)
        embedding = self.norm_embedding(embedding)
        embedding = self.dropout_embedding(embedding)
        return EmbeddedElementsSequence(embedding)
