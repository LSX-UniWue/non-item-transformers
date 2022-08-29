import functools
from typing import Dict, Any, Optional

from asme.core.models.ubert4rec.components import UBERT4RecSequenceElementsRepresentationComponent
from asme.core.models.bert4rec.bert4rec_model import normal_initialize_weights
from asme.core.models.common.components.representation_modifier.ffn_modifier import \
    FFNSequenceRepresentationModifierComponent
from asme.core.models.ubert4rec.components import UserTransformerSequenceRepresentationComponent
from asme.core.models.common.layers.layers import PROJECT_TYPE_LINEAR, build_projection_layer
from asme.core.models.common.layers.transformer_layers import TransformerEmbedding
from asme.core.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.core.utils.hyperparameter_utils import save_hyperparameters
from asme.core.utils.inject import InjectVocabularySize, InjectTokenizers


class UBERT4RecModel(SequenceRecommenderModel):

    @save_hyperparameters
    def __init__(self,
                 transformer_hidden_size: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 item_vocab_size: InjectVocabularySize("item"),
                 max_seq_length: int,
                 transformer_dropout: float,
                 additional_attributes: Dict[str, Dict[str, Any]],
                 additional_tokenizers: InjectTokenizers(),
                 user_attributes: Dict[str, Dict[str, Any]],
                 positional_embedding: True,
                 segment_embedding: False,
                 embedding_pooling_type: str = None,
                 initializer_range: float = 0.02,
                 transformer_intermediate_size: Optional[int] = None,
                 transformer_attention_dropout: Optional[float] = None):

        # save for later call by the training module
        
        self.additional_userdata_keys = []
        self.additional_metadata_keys = []
        if user_attributes is not None:
            self.additional_metadata_keys = list(user_attributes.keys())
            self.additional_userdata_keys = list(user_attributes.keys())
            max_seq_length += 1
            
        if additional_attributes is not None:
            if user_attributes is not None:
                self.additional_metadata_keys = self.additional_metadata_keys + list(additional_attributes.keys())
            else:
                self.additional_metadata_keys = list(additional_attributes.keys())
                
        print("METADATA",self.additional_metadata_keys)
        print("USERDATA",self.additional_metadata_keys)



        # embedding will be normed and dropout after all embeddings are added to the representation
        sequence_embedding = TransformerEmbedding(item_vocab_size, max_seq_length, transformer_hidden_size, 0.0,
                                                  embedding_pooling_type=embedding_pooling_type,
                                                  norm_embedding=False, positional_embedding=positional_embedding)

        element_representation = UBERT4RecSequenceElementsRepresentationComponent(sequence_embedding,
                                                                                   transformer_hidden_size,
                                                                                   additional_attributes,
                                                                                   user_attributes,
                                                                                   additional_tokenizers,
                                                                                   segment_embedding,
                                                                                   dropout=transformer_dropout,
                                                                                   replace_first_item=False)
        sequence_representation = UserTransformerSequenceRepresentationComponent(transformer_hidden_size,
                                                                                 num_transformer_heads,
                                                                                 num_transformer_layers,
                                                                                 transformer_dropout,
                                                                                 user_attributes,
                                                                                 bidirectional=False,
                                                                                 transformer_attention_dropout=transformer_attention_dropout,
                                                                                 transformer_intermediate_size=transformer_intermediate_size,)

        transform_layer = FFNSequenceRepresentationModifierComponent(transformer_hidden_size)

        projection_layer = build_projection_layer(PROJECT_TYPE_LINEAR, transformer_hidden_size, item_vocab_size,
                                                  sequence_embedding.item_embedding.embedding)

        super().__init__(element_representation, sequence_representation, transform_layer, projection_layer)

        # FIXME: move init code
        self.apply(functools.partial(normal_initialize_weights, initializer_range=initializer_range))

    def required_metadata_keys(self):
        return self.additional_metadata_keys

    def optional_metadata_keys(self):
        return self.additional_userdata_keys
