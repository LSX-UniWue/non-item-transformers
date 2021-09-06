import functools
from typing import Dict, Any, Optional

from asme.models.kebert4rec.components import KeBERT4RecSequenceElementsRepresentationComponent
from asme.models.bert4rec.bert4rec_model import normal_initialize_weights
from asme.models.common.components.representation_modifier.ffn_modifier import \
    FFNSequenceRepresentationModifierComponent
from asme.models.bert4rec.components import BidirectionalTransformerSequenceRepresentationComponent
from asme.models.common.layers.layers import PROJECT_TYPE_LINEAR, build_projection_layer
from asme.models.common.layers.transformer_layers import TransformerEmbedding
from asme.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.utils.hyperparameter_utils import save_hyperparameters


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

        element_representation = KeBERT4RecSequenceElementsRepresentationComponent(sequence_embedding,
                                                                                   transformer_hidden_size,
                                                                                   additional_attributes,
                                                                                   dropout=transformer_dropout)
        sequence_representation = BidirectionalTransformerSequenceRepresentationComponent(transformer_hidden_size,
                                                                                          num_transformer_heads,
                                                                                          num_transformer_layers,
                                                                                          transformer_dropout,
                                                                                          transformer_attention_dropout,
                                                                                          transformer_intermediate_size)

        transform_layer = FFNSequenceRepresentationModifierComponent(transformer_hidden_size)

        projection_layer = build_projection_layer(PROJECT_TYPE_LINEAR, transformer_hidden_size, item_vocab_size,
                                                  sequence_embedding.item_embedding.embedding)

        super().__init__(element_representation, sequence_representation, transform_layer, projection_layer)

        # FIXME: move init code
        self.apply(functools.partial(normal_initialize_weights, initializer_range=initializer_range))

    def required_metadata_keys(self):
        return self.additional_metadata_keys
