from torch import nn

from asme.core.models.common.layers.layers import IdentitySequenceRepresentationModifierLayer, LinearProjectionLayer
from asme.core.models.common.layers.transformer_layers import TransformerEmbedding
from asme.core.models.content_bert4rec.components import ContextSequenceElementsRepresentationComponent, \
    PrependedTransformerSequenceRepresentationComponent, ContextSequenceRepresentationModifierComponent
from asme.core.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.core.tokenization.tokenizer import Tokenizer
from asme.core.tokenization.item_dictionary import SpecialValues
from asme.core.utils.hyperparameter_utils import save_hyperparameters
from asme.core.utils.inject import InjectTokenizers, inject, InjectTokenizer, InjectDictionaries
from typing import Dict, Any

from asme.data.datasets import ITEM_SEQ_ENTRY_NAME

prefusion = "prefusion"
postfusion = "postfusion"
sequence_prepend = "prepend"

class ContentSASRecModel(SequenceRecommenderModel):
    """
    Implementation of the "Self-Attentive Sequential Recommendation" paper.
    see https://doi.org/10.1109%2fICDM.2018.00035 for more details

    see https://github.com/kang205/SASRec for the original Tensorflow implementation
    """
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
                 positional_embedding: bool = True,
                 segment_embedding: bool = False,
                 embedding_pooling_type: str = None,
                 transformer_intermediate_size: int = None,
                 transformer_attention_dropout: float = None,
                 mode: str = "full",
                 ):


        # save for later call by the training module
        self.item_metadata_keys = []
        self.sequence_metadata_keys = []
        self.add_keys_to_metadata(item_attributes, self.item_metadata_keys)
        self.add_keys_to_metadata(sequence_attributes, self.item_metadata_keys)
        self.add_keys_to_metadata(sequence_attributes, self.sequence_metadata_keys)

        prefusion_attributes = item_attributes.get(prefusion, None)
        postfusion_attributes = item_attributes.get(postfusion, None)
        prepend_attributes = sequence_attributes.get(sequence_prepend, None)

        # embedding will be normed and dropout after all embeddings are added to the representation
        embedding_layer = TransformerEmbedding(
            item_voc_size=len(item_tokenizer),
            max_seq_len=max_seq_length,
            embedding_size=transformer_hidden_size,
            dropout=transformer_dropout,
            embedding_pooling_type=embedding_pooling_type,
            positional_embedding=positional_embedding
        )

        self.mode = mode

        if mode == "neg_sampling":
            #  use positive / negative sampling for training and evaluation as described in the original paper
            raise Exception(f"{mode} is not supported yet, only <full>")
        elif mode == "full":
            # compute a full ranking over all items as necessary with cross-entropy loss
            projection_layer = LinearProjectionLayer(transformer_hidden_size, len(item_tokenizer))
        else:
            raise Exception(f"{mode} is an unknown projection mode. Choose either <full> or <neg_sampling>.")


        element_representation = ContextSequenceElementsRepresentationComponent(embedding_layer,
                                                                                transformer_hidden_size,
                                                                                item_tokenizer,
                                                                                prefusion_attributes,
                                                                                prepend_attributes,
                                                                                attribute_tokenizers,
                                                                                vector_dictionaries,
                                                                                dropout=transformer_dropout,
                                                                                segment_embedding_active=segment_embedding)

        sequence_representation = PrependedTransformerSequenceRepresentationComponent(transformer_hidden_size,
                                                                                      num_transformer_heads,
                                                                                      num_transformer_layers,
                                                                                      transformer_dropout,
                                                                                      sequence_attributes,
                                                                                      bidirectional=False,
                                                                                      transformer_attention_dropout=transformer_attention_dropout,
                                                                                      transformer_intermediate_size=transformer_intermediate_size,)

        if postfusion_attributes is not None:
            modifier_layer = ContextSequenceRepresentationModifierComponent(transformer_hidden_size,
                                                                            item_tokenizer,
                                                                            postfusion_attributes,
                                                                            sequence_attributes,
                                                                            attribute_tokenizers,
                                                                            vector_dictionaries)
        else:
            modifier_layer = IdentitySequenceRepresentationModifierLayer()


        super().__init__(element_representation, sequence_representation, modifier_layer, projection_layer)

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

