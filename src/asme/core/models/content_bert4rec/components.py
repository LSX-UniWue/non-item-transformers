from typing import Dict, Any

from asme.core.models.common.layers.data.sequence import InputSequence, EmbeddedElementsSequence, \
    SequenceRepresentation, ModifiedSequenceRepresentation
from asme.core.models.common.layers.layers import SequenceElementsRepresentationLayer, \
    SequenceRepresentationModifierLayer
from asme.core.models.common.layers.transformer_layers import TransformerEmbedding
from asme.core.models.kebert4rec.layers import LinearUpscaler
from torch import nn

from asme.core.tokenization.vector_dictionary import VectorDictionary
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
                 attribute_tokenizers: Dict[str, Tokenizer],
                 vector_dictionaries: Dict[str, VectorDictionary],
                 dropout: float = 0.0
                 ):

        super().__init__()
        self.prefusion_attribute = prefusion_attributes
        self.item_embedding_layer = item_embedding_layer

        prefusion_attribute_embeddings = {}
        if prefusion_attributes is not None:
            for attribute_name, attribute_infos in prefusion_attributes.items():
                embedding = attribute_infos["embedding"]
                if embedding == "keys":
                    embedding_type = attribute_infos['embedding_type']
                    vocab_size = len(attribute_tokenizers["tokenizers." + attribute_name])
                    prefusion_attribute_embeddings[attribute_name] = _build_embedding_type(
                        embedding_type=embedding_type,
                        vocab_size=vocab_size,
                        hidden_size=transformer_hidden_size)
                elif embedding == "vector":
                    vector_dict = vector_dictionaries[attribute_name]
                    default = vector_dict.unk_value
                    prefusion_attribute_embeddings[attribute_name] = ContentVectorMaskAndScale(len(default),
                                                                                               transformer_hidden_size,
                                                                                               item_tokenizer.mask_token_id)

        self.prefusion_attribute_embeddings = nn.ModuleDict(prefusion_attribute_embeddings)
        self.dropout_embedding = nn.Dropout(dropout)
        self.norm_embedding = nn.LayerNorm(transformer_hidden_size)

    def forward(self, sequence: InputSequence) -> EmbeddedElementsSequence:
        embedding_sequence = self.item_embedding_layer(sequence)
        embedding = embedding_sequence.embedded_sequence
        for input_key, module in self.prefusion_attribute_embeddings.items():
            attribute_infos = self.prefusion_attribute[input_key]
            embedding_type = attribute_infos["embedding"]
            merge = attribute_infos["merge_function"]
            additional_metadata = sequence.get_attribute(input_key)

            if embedding_type == "keys":
                embedded_data = module(additional_metadata)
            elif embedding_type == "vector":
                embedded_data = module(content_sequence=additional_metadata, item_sequence=sequence.sequence)

            merge_function = _merge_function(merge)
            embedding = merge_function(embedding, embedded_data)

        embedding = self.norm_embedding(embedding)
        embedding = self.dropout_embedding(embedding)
        return EmbeddedElementsSequence(embedding)

class ContextSequenceRepresentationModifierComponent(SequenceRepresentationModifierLayer):
    """
    layer that applies a linear layer and a activation function
    """

    @save_hyperparameters
    def __init__(self,
                 feature_size: int,
                 item_tokenizer: Tokenizer,
                 postfusion_attributes: Dict[str, Dict[str, Any]],
                 attribute_tokenizers: Dict[str, Tokenizer],
                 vector_dictionaries: Dict[str, VectorDictionary]
                 ):
        super().__init__()

        self.postfusion_attributes = postfusion_attributes

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
            postfusion_embedded_sequence = merge_function(postfusion_embedded_sequence, embedded_data)

        transformation = self.transform(postfusion_embedded_sequence)
        return ModifiedSequenceRepresentation(transformation)


def _merge_function(merge):
    def add_merge(sequence, context):
        sequence += context
        return sequence

    def multiply_merge(sequence, context):
        sequence *= context
        return sequence

    return {"add": add_merge,
            "multiply": multiply_merge}[merge]
