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


class PreFusionContextSequenceElementsRepresentationComponent(SequenceElementsRepresentationLayer):

    @save_hyperparameters
    def __init__(self,
                 item_embedding_layer: TransformerEmbedding,
                 transformer_hidden_size: int,
                 item_tokenizer: Tokenizer,
                 prefusion_attributes: Dict[str, Dict[str, Any]],
                 additional_attributes_tokenizer: Dict[str, Tokenizer],
                 vector_attributes: Dict[str, Dict[str, Any]],
                 vector_dictionaries: Dict[str, VectorDictionary],
                 dropout: float = 0.0
                 ):

        super().__init__()
        self.item_embedding_layer = item_embedding_layer

        prefusion_attribute_embeddings = {}
        if prefusion_attributes is not None:
            for attribute_name, attribute_infos in prefusion_attributes.items():
                embedding_type = attribute_infos['embedding_type']
                vocab_size = len(additional_attributes_tokenizer["tokenizers." + attribute_name])
                prefusion_attribute_embeddings[attribute_name] = _build_embedding_type(embedding_type=embedding_type,
                                                                                        vocab_size=vocab_size,
                                                                                        hidden_size=transformer_hidden_size)

        self.prefusion_attribute_embeddings = nn.ModuleDict(prefusion_attribute_embeddings)
        vector_embedding = {}
        if vector_dictionaries is not None:
            for attribute_name, attributes_infos in vector_attributes.items():
                vector_dict = vector_dictionaries[attribute_name]
                default = vector_dict.unk_value
                vector_embedding[attribute_name] = ContentVectorMaskAndScale(len(default), transformer_hidden_size, item_tokenizer.mask_token_id)
        self.vector_embedding = nn.ModuleDict(vector_embedding)

        self.dropout_embedding = nn.Dropout(dropout)
        self.norm_embedding = nn.LayerNorm(transformer_hidden_size)

    def forward(self, sequence: InputSequence) -> EmbeddedElementsSequence:
        embedding_sequence = self.item_embedding_layer(sequence)
        embedding = embedding_sequence.embedded_sequence
        for input_key, module in self.prefusion_attribute_embeddings.items():
            additional_metadata = sequence.get_attribute(input_key)
            test = module(additional_metadata)
            embedding += test

        for input_key, module in self.vector_embedding.items():
            additional_metadata = sequence.get_attribute(input_key)
            test = module(content_sequence=additional_metadata, item_sequence=sequence.sequence)
            embedding += test

        embedding = self.norm_embedding(embedding)
        embedding = self.dropout_embedding(embedding)
        return EmbeddedElementsSequence(embedding)

class PostFusionContextSequenceRepresentationModifierComponent(SequenceRepresentationModifierLayer):
    """
    layer that applies a linear layer and a activation function
    """

    @save_hyperparameters
    def __init__(self,
                 feature_size: int,
                 postfusion_attributes: Dict[str, Dict[str, Any]],
                 additional_attributes_tokenizer: Dict[str, Tokenizer],
                 merge_function: str = "add"
                 ):
        super().__init__()

        self.merge_function = merge_function

        postfusion_attribute_embeddings = {}
        for attribute_name, attribute_infos in postfusion_attributes.items():
            embedding_type = attribute_infos['embedding_type']
            vocab_size = len(additional_attributes_tokenizer["tokenizers." + attribute_name])
            postfusion_attribute_embeddings[attribute_name] = _build_embedding_type(embedding_type=embedding_type,
                                                                                   vocab_size=vocab_size,
                                                                                   hidden_size=feature_size)
        self.postfusion_attribute_embeddings = nn.ModuleDict(postfusion_attribute_embeddings)


        self.transform = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.GELU(),
            nn.LayerNorm(feature_size)
        )

    def forward(self, sequence_representation: SequenceRepresentation) -> ModifiedSequenceRepresentation:
        postfusion_embedded_sequence = sequence_representation.encoded_sequence

        #Sum attribute embeddings
        context_embeddings = None
        for input_key, module in self.postfusion_attribute_embeddings.items():
            additional_metadata = sequence_representation.input_sequence.get_attribute(input_key)
            pf_attribute = module(additional_metadata)
            if context_embeddings is None:
                context_embeddings = pf_attribute
            else:
                context_embeddings += pf_attribute

        if self.merge_function == "add":
            postfusion_embedded_sequence +=context_embeddings
        if self.merge_function == "multiply":
            postfusion_embedded_sequence *= context_embeddings

        transformation = self.transform(postfusion_embedded_sequence)
        return ModifiedSequenceRepresentation(transformation)