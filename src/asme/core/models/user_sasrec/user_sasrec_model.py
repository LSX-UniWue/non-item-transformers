from torch import nn

from asme.core.models.common.layers.layers import IdentitySequenceRepresentationModifierLayer, LinearProjectionLayer
from asme.core.models.common.layers.transformer_layers import TransformerEmbedding
from asme.core.models.ubert4rec.components import UBERT4RecSequenceElementsRepresentationComponent
from asme.core.models.ubert4rec.components import UserTransformerSequenceRepresentationComponent
from asme.core.models.user_sasrec.components import UserSASRecProjectionComponent
from asme.core.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.core.utils.hyperparameter_utils import save_hyperparameters
from asme.core.utils.inject import InjectVocabularySize, InjectTokenizers
from typing import Dict, Any


class UserSASRecModel(SequenceRecommenderModel):
    """
    Implementation of the "Self-Attentive Sequential Recommendation" paper.
    see https://doi.org/10.1109%2fICDM.2018.00035 for more details

    see https://github.com/kang205/SASRec for the original Tensorflow implementation
    """

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
                 segment_embedding: False,
                 embedding_pooling_type: str = None,
                 transformer_intermediate_size: int = None,
                 transformer_attention_dropout: float = None,
                 mode: str = "neg_sampling", # alternative: "full"
                 positional_embedding: bool = True,
                 replace_first_item: bool = False
                 ):


        self.additional_userdata_keys = []
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
        embedding_layer = TransformerEmbedding(
            item_voc_size=item_vocab_size,
            max_seq_len=max_seq_length,
            embedding_size=transformer_hidden_size,
            dropout=transformer_dropout,
            embedding_pooling_type=embedding_pooling_type,
            positional_embedding=positional_embedding
        )

        self.mode = mode

        if mode == "neg_sampling":
            #  use positive / negative sampling for training and evaluation as described in the original paper
            projection_layer = UserSASRecProjectionComponent(embedding_layer)
        elif mode == "full":
            # compute a full ranking over all items as necessary with cross-entropy loss
            projection_layer = LinearProjectionLayer(transformer_hidden_size, item_vocab_size)
        else:
            raise Exception(f"{mode} is an unknown projection mode. Choose either <full> or <neg_sampling>.")


        element_representation = UBERT4RecSequenceElementsRepresentationComponent(embedding_layer,
                                                                                  transformer_hidden_size,
                                                                                  additional_attributes,
                                                                                  user_attributes,
                                                                                  additional_tokenizers,
                                                                                  segment_embedding,
                                                                                  dropout=transformer_dropout,
                                                                                  replace_first_item=replace_first_item)
        sequence_representation = UserTransformerSequenceRepresentationComponent(transformer_hidden_size,
                                                                                 num_transformer_heads,
                                                                                 num_transformer_layers,
                                                                                 transformer_dropout,
                                                                                 user_attributes,
                                                                                 bidirectional=False,
                                                                                 transformer_attention_dropout=transformer_attention_dropout,
                                                                                 transformer_intermediate_size=transformer_intermediate_size,
                                                                                 replace_first_item=replace_first_item)

        transform_layer = IdentitySequenceRepresentationModifierLayer()

        super().__init__(element_representation, sequence_representation, transform_layer, projection_layer)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initializes the weights of the layers """
        is_linear_layer = isinstance(module, nn.Linear)
        is_embedding_layer = isinstance(module, nn.Embedding)
        if is_linear_layer or is_embedding_layer:
            nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if is_linear_layer and module.bias is not None:
            module.bias.data.zero_()

    def required_metadata_keys(self):
        return self.additional_metadata_keys

    def optional_metadata_keys(self):
        return self.additional_userdata_keys
