import functools
from typing import Dict, Any, Optional

from torch import nn

from asme.core.init.factories.features.tokenizer_factory import TOKENIZERS_PREFIX
from asme.core.models.bert4rec.bert4rec_model import normal_initialize_weights

from asme.core.models.common.components.representation_modifier.ffn_modifier import \
    FFNSequenceRepresentationModifierComponent
from asme.core.models.common.layers.layers import PROJECT_TYPE_LINEAR, build_projection_layer, ProjectionLayer
from asme.core.models.common.layers.transformer_layers import TransformerEmbedding
from asme.core.models.content_bert4rec.components import ContextSequenceRepresentationModifierComponent, \
    ContextSequenceElementsRepresentationComponent, PrependedTransformerSequenceRepresentationComponent
from asme.core.models.non_item_models.components import NonItemSequenceElementsRepresentationComponent
from asme.core.models.non_item_models.non_item_sasrec_model import CategoryAndItemProjectionLayer, \
    CategoryAndItemReuseProjectionLayer
from asme.core.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.core.tokenization.tokenizer import Tokenizer
from asme.core.tokenization.item_dictionary import SpecialValues
from asme.core.utils.hyperparameter_utils import save_hyperparameters
from asme.core.utils.inject import InjectTokenizers, inject, InjectDictionaries, InjectTokenizer
from asme.data.datasets import ITEM_SEQ_ENTRY_NAME, KEY_DELIMITER

prefusion = "prefusion"
postfusion = "postfusion"
sequence_prepend = "prepend"


class NonItemBERT4RecModel(SequenceRecommenderModel):

    @inject(
        item_tokenizer=InjectTokenizer(ITEM_SEQ_ENTRY_NAME),
        attribute_tokenizers=InjectTokenizers(),
        vector_dictionaries=InjectDictionaries()
    )
    @save_hyperparameters
    def __init__(self,
                 transformer_hidden_size: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 item_tokenizer: Tokenizer,
                 max_seq_length: int,
                 transformer_dropout: float,
                 item_attributes: Dict[str, Dict[str, Any]] = None,
                 sequence_attributes: Dict[str, Dict[str, Any]] = None,
                 attribute_tokenizers: Dict[str, Tokenizer] = None,
                 vector_dictionaries: Dict[str, SpecialValues] = None,
                 loss_category: str = None,
                 item_id_type_settings: Dict[str, Any] = None,
                 positional_embedding: bool = True,
                 segment_embedding: bool = False,
                 embedding_pooling_type: str = None,
                 project_layer_type: str = 'linear',
                 initializer_range: float = 0.02,
                 transformer_intermediate_size: Optional[int] = None,
                 transformer_attention_dropout: Optional[float] = None):

        # save for later call by the training module
        self.item_metadata_keys = []
        self.sequence_metadata_keys = []
        self.add_keys_to_metadata(item_attributes, self.item_metadata_keys)
        self.add_keys_to_metadata(sequence_attributes, self.item_metadata_keys)
        self.add_keys_to_metadata(sequence_attributes, self.sequence_metadata_keys)
        self.item_metadata_keys.append(item_id_type_settings["name"])

        prefusion_attributes = None if item_attributes is None else item_attributes.get(prefusion, None)
        postfusion_attributes = None if item_attributes is None else item_attributes.get(postfusion, None)
        prepend_attributes = None if sequence_attributes is None else sequence_attributes.get(sequence_prepend, None)

        # embedding will be normed and dropout after all embeddings are added to the representation
        sequence_embedding = TransformerEmbedding(len(item_tokenizer), max_seq_length, transformer_hidden_size, 0.0,
                                                  embedding_pooling_type=embedding_pooling_type,
                                                  norm_embedding=False, positional_embedding=positional_embedding)

        element_representation = NonItemSequenceElementsRepresentationComponent(sequence_embedding,
                                                                                transformer_hidden_size,
                                                                                item_tokenizer,
                                                                                prefusion_attributes,
                                                                                prepend_attributes,
                                                                                attribute_tokenizers,
                                                                                vector_dictionaries,
                                                                                item_id_type_settings,
                                                                                dropout=transformer_dropout,
                                                                                segment_embedding_active=segment_embedding)

        sequence_representation = PrependedTransformerSequenceRepresentationComponent(transformer_hidden_size,
                                                                                      num_transformer_heads,
                                                                                      num_transformer_layers,
                                                                                      transformer_dropout,
                                                                                      sequence_attributes,
                                                                                      bidirectional=False,
                                                                                      transformer_attention_dropout=transformer_attention_dropout,
                                                                                      transformer_intermediate_size=transformer_intermediate_size, )

        if postfusion_attributes is not None:
            modifier_layer = ContextSequenceRepresentationModifierComponent(transformer_hidden_size,
                                                                            item_tokenizer,
                                                                            postfusion_attributes,
                                                                            sequence_attributes,
                                                                            attribute_tokenizers,
                                                                            vector_dictionaries)
        else:
            modifier_layer = FFNSequenceRepresentationModifierComponent(transformer_hidden_size)

        cat_embedding = element_representation.prefusion_attribute_embeddings[loss_category]
        projection_layer = build_projection_layer(project_layer_type, transformer_hidden_size, len(item_tokenizer),
                                                  sequence_embedding.item_embedding.embedding,
                                                  len(attribute_tokenizers.get(TOKENIZERS_PREFIX+KEY_DELIMITER+loss_category)),
                                                  cat_embedding)

        super().__init__(element_representation, sequence_representation, modifier_layer, projection_layer)

        # FIXME: move init code
        self.apply(functools.partial(normal_initialize_weights, initializer_range=initializer_range))

        # FIXME: move init code
        self.apply(functools.partial(normal_initialize_weights, initializer_range=initializer_range))

    def required_metadata_keys(self):
        return self.item_metadata_keys

    def optional_metadata_keys(self):
        return self.sequence_metadata_keys

    def add_keys_to_metadata(self, dictionary, metadata_keys):
        if dictionary is not None:
            if dictionary.get(prefusion):
                metadata_keys.extend(list(dictionary[prefusion].keys()))
            if dictionary.get(postfusion):
                metadata_keys.extend(list(dictionary[postfusion].keys()))
            if dictionary.get(sequence_prepend):
                metadata_keys.extend(list(dictionary[sequence_prepend].keys()))


def build_projection_layer(project_type: str,
                           transformer_hidden_size: int,
                           item_voc_size: int,
                           item_embedding: nn.Embedding,
                           cat_voc_size: int,
                           cat_embedding: nn.Embedding
                           ) -> ProjectionLayer:
    if project_type == PROJECT_TYPE_LINEAR:
        return CategoryAndItemProjectionLayer(transformer_hidden_size, item_voc_size, cat_voc_size)

    if project_type == 'transpose_embedding':
        return CategoryAndItemReuseProjectionLayer(item_embedding, cat_embedding, item_voc_size, cat_voc_size)

    raise KeyError(f'{project_type} invalid projection layer')
