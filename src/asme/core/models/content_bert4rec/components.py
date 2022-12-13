from typing import Dict, Any, Optional

from torch.nn import ModuleDict

from asme.core.models.common.layers.data.sequence import InputSequence, EmbeddedElementsSequence, \
    SequenceRepresentation, ModifiedSequenceRepresentation
from asme.core.models.common.layers.layers import SequenceElementsRepresentationLayer, \
    SequenceRepresentationModifierLayer, SequenceRepresentationLayer
from asme.core.models.common.layers.transformer_layers import TransformerEmbedding, TransformerLayer
from asme.core.models.content_bert4rec.layers import LinearUpscaler
from torch import nn

from asme.core.tokenization.item_dictionary import ItemDictionary
from asme.core.utils.hyperparameter_utils import save_hyperparameters
from asme.data.datasets.processors.tokenizer import Tokenizer

import torch


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


class ContentVectorMaskAndScale(nn.Module):

    @save_hyperparameters
    def __init__(self, input_size: int, embed_size: int, item_mask_token: int):
        super().__init__()
        self.item_mask_token = item_mask_token
        self.linear = nn.Linear(input_size, embed_size)
        self.trained_mask = nn.Parameter(torch.Tensor(input_size))
        self.embedding_norm = nn.LayerNorm(input_size)
        nn.init.normal_(self.trained_mask, mean=1, std=0.5)

    def forward(self,
                content_sequence: torch.Tensor,
                item_sequence: torch.Tensor
                ) -> torch.Tensor:
        mask_indices = (item_sequence == self.item_mask_token).unsqueeze(-1)
        sequence = torch.where(mask_indices, self.trained_mask, content_sequence)
        sequence = self.embedding_norm(sequence)
        sequence = self.linear(sequence)

        return sequence


class ContextSequenceElementsRepresentationComponent(SequenceElementsRepresentationLayer):

    @save_hyperparameters
    def __init__(self,
                 item_embedding_layer: TransformerEmbedding,
                 transformer_hidden_size: int,
                 item_tokenizer: Tokenizer,
                 prefusion_attributes: Dict[str, Dict[str, Any]],
                 sequence_attributes: Dict[str, Dict[str, Any]],
                 attribute_tokenizers: Dict[str, Tokenizer],
                 vector_dictionaries: Dict[str, ItemDictionary],
                 dropout: float = 0.0,
                 segment_embedding_active: bool = False
                 ):

        super().__init__()
        self.prefusion_attributes_dict = prefusion_attributes
        self.sequence_attributes_dict = sequence_attributes
        self.item_embedding_layer = item_embedding_layer
        self.use_segment_embedding = segment_embedding_active

        prefusion_attribute_embeddings = {}
        if prefusion_attributes is not None:
            self.create_attribute_embeddings(attribute_tokenizers, item_tokenizer, prefusion_attribute_embeddings,
                                             prefusion_attributes, transformer_hidden_size, vector_dictionaries)
        sequence_attribute_embeddings = {}
        if sequence_attributes is not None:
            self.create_attribute_embeddings(attribute_tokenizers, item_tokenizer, sequence_attribute_embeddings,
                                             sequence_attributes, transformer_hidden_size, vector_dictionaries)

        self.prefusion_attribute_embeddings = nn.ModuleDict(prefusion_attribute_embeddings)

        self.sequence_attribute_embeddings = nn.ModuleDict(sequence_attribute_embeddings)

        if self.use_segment_embedding == True:
            self.segment_embedding = nn.Embedding(self.attribute_types,
                                                  len(prefusion_attributes) + len(sequence_attributes))

        self.dropout_embedding = nn.Dropout(dropout)
        self.norm_embedding = nn.LayerNorm(transformer_hidden_size)

    @staticmethod
    def create_attribute_embeddings(attribute_tokenizers, item_tokenizer, attribute_embeddings,
                                    attributes, transformer_hidden_size, vector_dictionaries):
        for attribute_name, attribute_infos in attributes.items():
            embedding = attribute_infos["embedding"]
            if embedding == "keys":
                embedding_type = attribute_infos['embedding_type']
                vocab_size = len(attribute_tokenizers["tokenizers." + attribute_name])
                attribute_embeddings[attribute_name] = _build_embedding_type(
                    embedding_type=embedding_type,
                    vocab_size=vocab_size,
                    hidden_size=transformer_hidden_size)
            elif embedding == "vector":
                vector_dict = vector_dictionaries[attribute_name]
                default = vector_dict.unk_value
                attribute_embeddings[attribute_name] = ContentVectorMaskAndScale(len(default),
                                                                                 transformer_hidden_size,
                                                                                 item_tokenizer.mask_token_id)

    def forward(self, sequence: InputSequence) -> EmbeddedElementsSequence:
        embedding_sequence = self.item_embedding_layer(sequence)
        embedding = embedding_sequence.embedded_sequence

        for input_key, module in self.prefusion_attribute_embeddings.items():
            embedded_data, merge_function = self.get_embedded_attribute(self.prefusion_attributes_dict, input_key,
                                                                        module, sequence)
            embedding = merge_function(embedding, embedded_data)

        sequence_metadata_embedding = None
        for input_key, module in self.sequence_attribute_embeddings.items():
            embedded_data, merge_function = self.get_embedded_attribute(self.sequence_attributes_dict, input_key,
                                                                        module, sequence)
            embedded_data = embedded_data[:, 0:1]

            if sequence_metadata_embedding is not None:
                sequence_metadata_embedding = merge_function(sequence_metadata_embedding, embedded_data)
            else:
                sequence_metadata_embedding = embedded_data

        if sequence_metadata_embedding is not None:
            embedding = torch.cat([sequence_metadata_embedding, embedding], dim=1)

        if self.use_segment_embedding:
            segments = torch.ones(sequence.sequence.shape, dtype=torch.int64, device=sequence.sequence.device)
            if sequence_metadata_embedding is not None:
                user_segment = torch.zeros(sequence.sequence.shape[0], 1, dtype=torch.int64,
                                           device=sequence.sequence.device)
                segments = torch.cat([user_segment, segments], dim=1)
            embedding += self.segment_embedding(segments)

        embedding = self.norm_embedding(embedding)
        embedding = self.dropout_embedding(embedding)
        return EmbeddedElementsSequence(embedding)

    def get_embedded_attribute(self, attribute, input_key, module, sequence):
        attribute_infos = attribute[input_key]
        embedding_type = attribute_infos["embedding"]
        merge = attribute_infos.get("merge_function", None)
        additional_metadata = sequence.get_attribute(input_key)
        if embedding_type == "keys":
            embedded_data = module(additional_metadata)
        elif embedding_type == "vector":
            embedded_data = module(content_sequence=additional_metadata, item_sequence=sequence.sequence)
        merge_function = _merge_function(merge)
        return embedded_data, merge_function


class ContextSequenceRepresentationModifierComponent(SequenceRepresentationModifierLayer):
    """
    layer that applies a linear layer and a activation function
    """

    @save_hyperparameters
    def __init__(self,
                 feature_size: int,
                 item_tokenizer: Tokenizer,
                 postfusion_attributes: Dict[str, Dict[str, Any]],
                 sequence_attributes: Dict[str, Dict[str, Any]],
                 attribute_tokenizers: Dict[str, Tokenizer],
                 vector_dictionaries: Dict[str, ItemDictionary],
                 use_transform_layer: bool = True
                 ):
        super().__init__()

        self.use_transform_layer = use_transform_layer
        self.postfusion_attributes = postfusion_attributes
        self.added_sequence_attributes = sequence_attributes is not None

        postfusion_attribute_embeddings = {}
        if postfusion_attributes is not None:
            for attribute_name, attribute_infos in postfusion_attributes.items():
                embedding = attribute_infos["embedding"]
                if embedding == "keys":
                    embedding_type = attribute_infos['embedding_type']
                    vocab_size = len(attribute_tokenizers["tokenizers." + attribute_name])
                    postfusion_attribute_embeddings[attribute_name] = _build_embedding_type(
                        embedding_type=embedding_type,
                        vocab_size=vocab_size,
                        hidden_size=feature_size)
                elif embedding == "vector":
                    vector_dict = vector_dictionaries[attribute_name]
                    default = vector_dict.unk_value
                    postfusion_attribute_embeddings[attribute_name] = ContentVectorMaskAndScale(len(default),
                                                                                                feature_size,
                                                                                                item_tokenizer.mask_token_id)

        self.postfusion_attribute_embeddings = nn.ModuleDict(postfusion_attribute_embeddings)

        if self.use_transform_layer:
            self.transform = nn.Sequential(
                nn.Linear(feature_size, feature_size),
                nn.GELU(),
                nn.LayerNorm(feature_size)
            )

    def forward(self, sequence_representation: SequenceRepresentation) -> ModifiedSequenceRepresentation:
        postfusion_embedded_sequence = sequence_representation.encoded_sequence

        for input_key, module in self.postfusion_attribute_embeddings.items():
            attribute_infos = self.postfusion_attributes[input_key]
            embedding_type = attribute_infos["embedding"]
            merge = attribute_infos["merge_function"]
            additional_metadata = sequence_representation.input_sequence.get_attribute(input_key)

            if embedding_type == "keys":
                embedded_data = module(additional_metadata)
            elif embedding_type == "vector":
                embedded_data = module(content_sequence=additional_metadata,
                                       item_sequence=sequence_representation.input_sequence.sequence)

            merge_function = _merge_function(merge)

            if self.added_sequence_attributes:
                postfusion_embedded_sequence[:, 1:, :] = merge_function(postfusion_embedded_sequence[:, 1:, :],
                                                                        embedded_data)

            else:
                postfusion_embedded_sequence = merge_function(postfusion_embedded_sequence, embedded_data)

        if self.use_transform_layer:
            transformation = self.transform(postfusion_embedded_sequence)
        else:
            transformation = postfusion_embedded_sequence
        return ModifiedSequenceRepresentation(transformation)


class PrependedTransformerSequenceRepresentationComponent(SequenceRepresentationLayer):
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
                 transformer_intermediate_size: Optional[int] = None):
        super().__init__()
        self.user_attributes = user_attributes
        self.bidirectional = bidirectional
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


def _merge_function(merge):
    def add_merge(sequence, context):
        sequence += context
        return sequence

    def multiply_merge(sequence, context):
        sequence *= context
        return sequence

    return {"add": add_merge,
            "multiply": multiply_merge}.get(merge, None)
