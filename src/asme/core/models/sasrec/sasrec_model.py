from typing import Dict, Any

from torch import nn

from asme.core.models.common.layers.layers import IdentitySequenceRepresentationModifierLayer, LinearProjectionLayer
from asme.core.models.common.layers.transformer_layers import TransformerEmbedding
from asme.core.models.kebert4rec.components import PreFusionContextSequenceElementsRepresentationComponent
from asme.core.models.sasrec.components import SASRecProjectionComponent, \
    PostFusionIdentitySequenceRepresentationModifierLayer
from asme.core.models.transformer.transformer_encoder_model import TransformerEncoderModel
from asme.core.tokenization.tokenizer import Tokenizer
from asme.core.tokenization.item_dictionary import ItemDictionary
from asme.core.utils.hyperparameter_utils import save_hyperparameters
from asme.core.utils.inject import InjectVocabularySize, inject, InjectTokenizers, InjectTokenizer


class SASRecModel(TransformerEncoderModel):
    """
    Implementation of the "Self-Attentive Sequential Recommendation" paper.
    see https://doi.org/10.1109%2fICDM.2018.00035 for more details

    see https://github.com/kang205/SASRec for the original Tensorflow implementation
    """

    @inject(
        item_tokenizer=InjectTokenizer("item"),
        item_vocab_size=InjectVocabularySize("item"),
        additional_attributes_tokenizer=InjectTokenizers()
    )
    @save_hyperparameters
    def __init__(self,
                 transformer_hidden_size: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 item_vocab_size: int,
                 max_seq_length: int,
                 transformer_dropout: float,
                 item_tokenizer: Tokenizer,
                 vector_dictionaries: Dict[str, ItemDictionary] = None,
                 prefusion_attributes: Dict[str, Dict[str, Any]] = None,
                 postfusion_attributes:  Dict[str, Dict[str, Any]] = None,
                 additional_attributes_tokenizer: Dict[str, Tokenizer] = None,
                 postfusion_merge_function: str = "add",
                 embedding_pooling_type: str = None,
                 transformer_intermediate_size: int = None,
                 transformer_attention_dropout: float = None,
                 mode: str = "neg_sampling"  # alternative: "full"
                 ):
        self.additional_metadata_keys = list()
        self.add_keys_to_metadata_keys(prefusion_attributes)
        self.add_keys_to_metadata_keys(postfusion_attributes)
        self.add_keys_to_metadata_keys(vector_dictionaries)

        sequence_embedding_layer = TransformerEmbedding(
            item_voc_size=item_vocab_size,
            max_seq_len=max_seq_length,
            embedding_size=transformer_hidden_size,
            dropout=transformer_dropout,
            embedding_pooling_type=embedding_pooling_type,
            positional_embedding=True
        )

        element_representation = PreFusionContextSequenceElementsRepresentationComponent(sequence_embedding_layer,
                                                                                         transformer_hidden_size,
                                                                                         item_tokenizer,
                                                                                         prefusion_attributes,
                                                                                         additional_attributes_tokenizer,
                                                                                         vector_dictionaries,
                                                                                         dropout=transformer_dropout)

        self.mode = mode

        if mode == "neg_sampling":
            #  use positive / negative sampling for training and evaluation as described in the original paper
            projection_layer = SASRecProjectionComponent(sequence_embedding_layer)
        elif mode == "full":
            # compute a full ranking over all items as necessary with cross-entropy loss
            projection_layer = LinearProjectionLayer(transformer_hidden_size, item_vocab_size)
        else:
            raise Exception(f"{mode} is an unknown projection mode. Choose either <full> or <neg_sampling>.")

        if postfusion_attributes is not None:
            modifier_layer = PostFusionIdentitySequenceRepresentationModifierLayer(transformer_hidden_size,
                                                                                  postfusion_attributes,
                                                                                  additional_attributes_tokenizer,
                                                                                  postfusion_merge_function)
        else:
            modifier_layer = IdentitySequenceRepresentationModifierLayer()

        super().__init__(
            transformer_hidden_size=transformer_hidden_size,
            num_transformer_heads=num_transformer_heads,
            num_transformer_layers=num_transformer_layers,
            transformer_dropout=transformer_dropout,
            bidirectional=False,
            embedding_layer=element_representation,
            sequence_representation_modifier_layer=modifier_layer,
            projection_layer=projection_layer,
            transformer_intermediate_size=transformer_intermediate_size,
            transformer_attention_dropout=transformer_attention_dropout
        )

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

    def add_keys_to_metadata_keys(self, dictionary):
        if dictionary is not None:
            self.additional_metadata_keys.extend(list(dictionary.keys()))
        print("Add",self.additional_metadata_keys)

    def required_metadata_keys(self):
        return self.additional_metadata_keys
