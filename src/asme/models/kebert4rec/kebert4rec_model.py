import functools
from typing import Dict, Any, Optional
from torch import nn

import torch

from asme.models.bert4rec.bert4rec_model import BidirectionalTransformerSequenceRepresentationLayer, \
    FFNSequenceRepresentationModifierLayer, normal_initialize_weights
from asme.models.common.layers.data.sequence import InputSequence, EmbeddedElementsSequence
from asme.models.common.layers.layers import PROJECT_TYPE_LINEAR, build_projection_layer
from asme.models.common.layers.transformer_layers import TransformerEmbedding
from asme.models.sequence_recommendation_model import SequenceElementsRepresentationLayer, SequenceRecommenderModel
from asme.utils.hyperparameter_utils import save_hyperparameters


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


class LinearUpscaler(nn.Module):

    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()
        self.linear = nn.Linear(vocab_size, embed_size)
        self.vocab_size = vocab_size

    def forward(self,
                content_input: torch.Tensor
                ) -> torch.Tensor:
        """
        :param content_input: a tensor containing the ids of each
        :return:
        """
        # the input is a sequence of content ids without any order
        # so we convert them into a multi-hot encoding
        multi_hot = torch.nn.functional.one_hot(content_input, self.vocab_size).sum(2).float()
        # 0 is the padding category, so zero it out
        multi_hot[:, :, 0] = 0
        return self.linear(multi_hot)


class KeBERT4RecSequenceElementsRepresentationLayer(SequenceElementsRepresentationLayer):

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


class KeBERT4RecModel(SequenceRecommenderModel):

    @save_hyperparameters
    def __init__(self,
                 transformer_hidden_size: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 item_vocab_size: int,
                 max_seq_length: int,
                 transformer_dropout: float,
                 additional_attributes: Dict[str, Dict[str, Any]],
                 embedding_pooling_type: str = None,
                 initializer_range: float = 0.02,
                 transformer_intermediate_size: Optional[int] = None,
                 transformer_attention_dropout: Optional[float] = None):

        # save for later call by the training module
        self.additional_metadata_keys = list(additional_attributes.keys())

        # embedding will be normed and dropout after all embeddings are added to the representation
        sequence_embedding = TransformerEmbedding(item_vocab_size, max_seq_length, transformer_hidden_size, 0.0,
                                                  embedding_pooling_type=embedding_pooling_type,
                                                  norm_embedding=False)

        element_representation = KeBERT4RecSequenceElementsRepresentationLayer(sequence_embedding,
                                                                               transformer_hidden_size,
                                                                               additional_attributes,
                                                                               dropout=transformer_dropout)
        sequence_representation = BidirectionalTransformerSequenceRepresentationLayer(transformer_hidden_size,
                                                                                      num_transformer_heads,
                                                                                      num_transformer_layers,
                                                                                      transformer_dropout,
                                                                                      transformer_attention_dropout,
                                                                                      transformer_intermediate_size)

        transform_layer = FFNSequenceRepresentationModifierLayer(transformer_hidden_size)

        projection_layer = build_projection_layer(PROJECT_TYPE_LINEAR, transformer_hidden_size, item_vocab_size,
                                                  sequence_embedding.item_embedding.embedding)

        super().__init__(element_representation, sequence_representation, transform_layer, projection_layer)

        # FIXME: move init code
        self.apply(functools.partial(normal_initialize_weights, initializer_range=initializer_range))

    def required_metadata_keys(self):
        return self.additional_metadata_keys
