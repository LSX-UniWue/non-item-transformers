from typing import Dict, Any

import torch
from torch import nn

from asme.core.models.common.layers.data.sequence import InputSequence, \
    EmbeddedElementsSequence
from asme.core.models.common.layers.layers import SequenceElementsRepresentationLayer
from asme.core.models.common.layers.transformer_layers import TransformerEmbedding
from asme.core.models.content_bert4rec.components import ContextSequenceElementsRepresentationComponent, \
    _merge_function, _build_embedding_type, ContentVectorMaskAndScale
from asme.core.tokenization.special_values import SpecialValues
from asme.core.tokenization.tokenizer import Tokenizer
from asme.core.utils.hyperparameter_utils import save_hyperparameters


class NonItemSequenceElementsRepresentationComponent(SequenceElementsRepresentationLayer):

    @save_hyperparameters
    def __init__(self, item_embedding_layer: TransformerEmbedding, transformer_hidden_size: int,
                 item_tokenizer: Tokenizer, prefusion_attributes: Dict[str, Dict[str, Any]],
                 sequence_attributes: Dict[str, Dict[str, Any]], attribute_tokenizers: Dict[str, Tokenizer],
                 vector_dictionaries: Dict[str, SpecialValues],
                 item_id_type_settings: Dict[str, Any] = None, dropout: float = 0.0,
                 no_id_for_plp: bool = False,
                 segment_embedding_active: bool = False):

        super().__init__()
        self.prefusion_attributes_dict = prefusion_attributes
        self.sequence_attributes_dict = sequence_attributes
        self.item_embedding_layer = item_embedding_layer
        self.use_segment_embedding = segment_embedding_active
        self.item_id_type_settings = item_id_type_settings
        self.no_id_for_plp = no_id_for_plp


        self.item_id_type_extra_embbedding = self.item_id_type_settings.get("extra_embbedding", None)
        if self.item_id_type_extra_embbedding:
            self.item_id_type_embedding = _build_embedding_type(
                embedding_type='content_embedding',
                vocab_size=2,
                hidden_size=transformer_hidden_size)

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
                vector_dict = vector_dictionaries["special_values."+attribute_name]
                default = vector_dict.unk_value
                attribute_embeddings[attribute_name] = ContentVectorMaskAndScale(len(default),
                                                                                 transformer_hidden_size,
                                                                                 item_tokenizer.mask_token_id)

    def forward(self, sequence: InputSequence) -> EmbeddedElementsSequence:
        embedding_sequence = self.item_embedding_layer(sequence)
        embedding = embedding_sequence.embedded_sequence

        if self.no_id_for_plp:
            no_id_embedding = torch.zeros(embedding.shape)
            type_info_tensor = sequence.get_attribute(self.item_id_type_settings["name"])
            type_info_tensor = type_info_tensor.unsqueeze(2)
            type_info_tensor = type_info_tensor.repeat(1,1,embedding.shape[2]) > 0
            embedding = torch.where(type_info_tensor,embedding, no_id_embedding)

        #Add the item /non-item embedding
        if self.item_id_type_extra_embbedding:
            additional_metadata = sequence.get_attribute(self.item_id_type_settings["name"])
            embedded_data = self.item_id_type_embedding(additional_metadata)
            merge_function = _merge_function(self.item_id_type_settings["merge"])
            embedding = merge_function(embedded_data, embedding)

        #Add item attributes
        for input_key, module in self.prefusion_attribute_embeddings.items():
            embedded_data, merge_function = self.get_embedded_attribute(self.prefusion_attributes_dict, input_key,
                                                                        module, sequence)
            embedding = merge_function(embedding, embedded_data)

        #User/sequence attribbutes
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

        #User segmeent embedding
        if self.use_segment_embedding:
            segments = torch.ones(sequence.sequence.shape, dtype=torch.int64, device=self.device)
            if sequence_metadata_embedding is not None:
                user_segment = torch.zeros(sequence.sequence.shape[0], 1, dtype=torch.int64,
                                           device=self.device)
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
        skip_item = attribute_infos.get("plp_only", False)
        if skip_item:
            item_id_type = sequence.get_attribute(self.item_id_type_settings["name"])
            pad_tensor = torch.zeros(additional_metadata.size()[2], device=sequence.sequence.device).type(additional_metadata.dtype)
            additional_metadata[item_id_type == 1] = pad_tensor




        if embedding_type == "keys":
            embedded_data = module(additional_metadata)
        elif embedding_type == "vector":
            embedded_data = module(content_sequence=additional_metadata, item_sequence=sequence.sequence)
        merge_function = _merge_function(merge)
        return embedded_data, merge_function
