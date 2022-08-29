from typing import Dict, Any, Optional
import torch

from asme.core.models.common.layers.data.sequence import InputSequence, EmbeddedElementsSequence, SequenceRepresentation
from asme.core.models.common.layers.layers import SequenceElementsRepresentationLayer, SequenceRepresentationLayer
from asme.core.models.common.layers.transformer_layers import TransformerEmbedding,TransformerLayer
from asme.core.models.kebert4rec.layers import LinearUpscaler
from torch import nn
from asme.data.datasets.processors.tokenizer import Tokenizer


class UserLinearUpscaler(nn.Module):

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
        # 0 is the padding category, so zero it out, but not for the user
        #multi_hot[:, :, 0] = 0
        return self.linear(multi_hot)

def _build_embedding_type(embedding_type: str,
                          vocab_size: int,
                          hidden_size: int
                          ) -> nn.Module:
    return {
        'user_embedding': nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=hidden_size),
        'content_embedding': nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=hidden_size),
        'linear_upscale': LinearUpscaler(vocab_size=vocab_size,
                                         embed_size=hidden_size),
        'user_linear_upscale': UserLinearUpscaler(vocab_size=vocab_size,
                                         embed_size=hidden_size)
    }[embedding_type]


class UBERT4RecSequenceElementsRepresentationComponent(SequenceElementsRepresentationLayer):

    def __init__(self,
                 item_embedding_layer: TransformerEmbedding,
                 embedding_size: int,
                 additional_attributes: Dict[str, Dict[str, Any]],
                 user_attributes: Dict[str, Dict[str, Any]],
                 additional_tokenizers: Dict[str, Tokenizer],
                 segment_embedding: bool = True,
                 dropout: float = 0.0,
                 replace_first_item: bool = False
                 ):
        super().__init__()

        self.segment_embedding = None
        self.item_embedding_layer = item_embedding_layer
        self.segment_embedding_active = segment_embedding
        self.attribute_types = 0
        self.replace_first_item = replace_first_item

        additional_attribute_embeddings = {}
        user_attribute_embeddings = {}
        if additional_attributes is not None:
            for attribute_name, attribute_infos in additional_attributes.items():
                embedding_type = attribute_infos['embedding_type']
                vocab_size = len(additional_tokenizers["tokenizers." + attribute_name])
                additional_attribute_embeddings[attribute_name] = _build_embedding_type(embedding_type=embedding_type,
                                                                                    vocab_size=vocab_size,
                                                                                    hidden_size=embedding_size)
            self.attribute_types += 1
        self.additional_attribute_embeddings = nn.ModuleDict(additional_attribute_embeddings)
        if user_attributes is not None:
            for attribute_name, attribute_infos in user_attributes.items():
                embedding_type = attribute_infos['embedding_type']
                vocab_size = len(additional_tokenizers["tokenizers." + attribute_name])
                user_attribute_embeddings[attribute_name] = _build_embedding_type(embedding_type=embedding_type,
                                                                                        vocab_size=vocab_size,
                                                                                        hidden_size=embedding_size)
                self.attribute_types += 1
        self.user_attribute_embeddings = nn.ModuleDict(user_attribute_embeddings)

        if self.segment_embedding_active == True:
            self.segment_embedding = nn.Embedding(self.attribute_types, embedding_size)

        self.dropout_embedding = nn.Dropout(dropout)
        self.norm_embedding = nn.LayerNorm(embedding_size)

    def forward(self, sequence: InputSequence) -> EmbeddedElementsSequence:
        embedding_sequence = self.item_embedding_layer(sequence)
        embedding = embedding_sequence.embedded_sequence

        user_embedding = None

        for input_key, module in self.additional_attribute_embeddings.items():
            additional_metadata = sequence.get_attribute(input_key)
            t = module(additional_metadata)
            embedding += t

        for input_key, module in self.user_attribute_embeddings.items():
            user_metadata = sequence.get_attribute(input_key)
            user_metadata = user_metadata[:, 0:1]

            if user_embedding != None:
                user_embedding += module(user_metadata)
            else:
                user_embedding = module(user_metadata)

        if user_embedding is not None:
            if self.replace_first_item:
                embedding = embedding[:,1:,:]
                embedding = torch.cat([user_embedding,embedding], dim=1)
            else:
                embedding = torch.cat([user_embedding,embedding], dim=1)

        if self.segment_embedding_active:
            segments = torch.ones(sequence.sequence.shape, dtype=torch.int64, device=sequence.sequence.device)
            if user_embedding is not None:
                user_segment = torch.zeros(sequence.sequence.shape[0], 1, dtype=torch.int64, device=sequence.sequence.device)
                segments = torch.cat([user_segment, segments], dim=1)
            embedding += self.segment_embedding(segments)

        embedding = self.norm_embedding(embedding)
        embedding = self.dropout_embedding(embedding)

        return EmbeddedElementsSequence(embedding, input_sequence=sequence)

class UserTransformerSequenceRepresentationComponent(SequenceRepresentationLayer):
    """
    A representation layer that uses a bidirectional transformer layer(s) to encode the given sequence
    """

    def __init__(self,
                 transformer_hidden_size: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 transformer_dropout: float,
                 user_attributes: Dict[str, Dict[str, Any]],
                 bidirectional: bool,
                 transformer_attention_dropout: Optional[float] = None,
                 transformer_intermediate_size: Optional[int] = None,
                 replace_first_item: bool = False):
        super().__init__()
        self.user_attributes = user_attributes
        self.bidirectional = bidirectional
        self.replace_first_item = replace_first_item
        if transformer_intermediate_size is None:
            transformer_intermediate_size = 4 * transformer_hidden_size

        self.transformer_encoder = TransformerLayer(transformer_hidden_size, num_transformer_heads,
                                                    num_transformer_layers, transformer_intermediate_size,
                                                    transformer_dropout,
                                                    attention_dropout=transformer_attention_dropout)

    def forward(self, embedded_sequence: EmbeddedElementsSequence) -> SequenceRepresentation:
        sequence = embedded_sequence.embedded_sequence
        padding_mask = embedded_sequence.input_sequence.padding_mask

        input_size = sequence.size()
        batch_size = input_size[0]
        sequence_length = sequence.size()[1]

        if padding_mask is not None:
            if len(self.user_attributes):
                if self.replace_first_item == False:
                    pad_user = torch.ones((padding_mask.shape[0], 1), device=sequence.device)
                    padding_mask = torch.cat([pad_user, padding_mask], dim=1)

        """ 
        We have to distinguish 4 cases here:
            - Bidirectional and no padding mask: Transformer can attend to all tokens with no restrictions
            - Bidirectional and padding mask: Transformer can attend to all tokens but those marked with the padding 
              mask
            - Unidirectional and no padding mask: Transformer can attend to all tokens up to the current sequence index
            - Unidirectional and padding mask: Transformer can attend to all tokens up to the current sequence index
              except those marked by the padding mask
        """

        if self.bidirectional:
            if padding_mask is None:
                attention_mask = None
            else:
                attention_mask = padding_mask.unsqueeze(1).repeat(1, sequence_length, 1).unsqueeze(1)
        else:
            if padding_mask is None:
                attention_mask = torch.tril(
                    torch.ones([sequence_length, sequence_length], device=sequence.device)).unsqueeze(0).repeat(
                    batch_size, 1, 1).unsqueeze(1)

            else:
                attention_mask = torch.tril(
                    torch.ones([sequence_length, sequence_length], device=sequence.device)).unsqueeze(0).repeat(
                    batch_size, 1, 1).unsqueeze(1)
                attention_mask *= padding_mask.unsqueeze(1).repeat(1, sequence_length, 1).unsqueeze(1)

        encoded_sequence = self.transformer_encoder(sequence, attention_mask=attention_mask)
        return SequenceRepresentation(encoded_sequence)