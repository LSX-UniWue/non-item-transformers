import math
import torch
from torch import nn

from asme.core.init.factories.features.tokenizer_factory import TOKENIZERS_PREFIX
from asme.core.models.common.layers.data.sequence import ModifiedSequenceRepresentation
from asme.core.models.common.layers.layers import IdentitySequenceRepresentationModifierLayer, LinearProjectionLayer, \
    ProjectionLayer
from asme.core.models.common.layers.transformer_layers import TransformerEmbedding
from asme.core.models.content_bert4rec.components import ContextSequenceElementsRepresentationComponent, \
    PrependedTransformerSequenceRepresentationComponent, ContextSequenceRepresentationModifierComponent
from asme.core.models.non_item_models.components import NonItemSequenceElementsRepresentationComponent
from asme.core.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.core.modules.util.module_util import convert_target_to_multi_hot
from asme.core.tokenization.tokenizer import Tokenizer
from asme.core.tokenization.item_dictionary import SpecialValues
from asme.core.utils.hyperparameter_utils import save_hyperparameters
from asme.core.utils.inject import InjectTokenizers, inject, InjectTokenizer, InjectDictionaries
from typing import Dict, Any, Union, Tuple

from asme.data.datasets import ITEM_SEQ_ENTRY_NAME
from asme.data.datasets.processors.utils import random_uniform

prefusion = "prefusion"
postfusion = "postfusion"
sequence_prepend = "prepend"


class NonItemSASRecModel(SequenceRecommenderModel):
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
    def __init__(self, transformer_hidden_size: int, num_transformer_heads: int, num_transformer_layers: int,
                 item_tokenizer: Tokenizer, max_seq_length: int, transformer_dropout: float,
                 item_attributes: Dict[str, Dict[str, Any]] = None,
                 sequence_attributes: Dict[str, Dict[str, Any]] = None,
                 attribute_tokenizers: Dict[str, Tokenizer] = None,
                 vector_dictionaries: Dict[str, SpecialValues] = None,
                 loss_category: str = None,
                 item_id_type_settings: Dict[str, Any] = None, positional_embedding: bool = True,
                 segment_embedding: bool = False, embedding_pooling_type: str = None,
                 transformer_intermediate_size: int = None, transformer_attention_dropout: float = None,
                 linked_projection_layer: int = 0):

        # save for later call by the training module
        self.item_metadata_keys = []
        self.sequence_metadata_keys = []
        self._add_keys_to_metadata(item_attributes, self.item_metadata_keys)
        self._add_keys_to_metadata(sequence_attributes, self.item_metadata_keys)
        self._add_keys_to_metadata(sequence_attributes, self.sequence_metadata_keys)
        self.item_metadata_keys.append(item_id_type_settings["name"])
        self.item_metadata_keys.append(loss_category+".target")

        prefusion_attributes = get_attributes(item_attributes, prefusion)
        postfusion_attributes = get_attributes(item_attributes, postfusion)
        prepend_attributes = get_attributes(sequence_attributes, sequence_prepend)

        # embedding will be normed and dropout after all embeddings are added to the representation
        embedding_layer = TransformerEmbedding(
            item_voc_size=len(item_tokenizer),
            max_seq_len=max_seq_length,
            embedding_size=transformer_hidden_size,
            dropout=transformer_dropout,
            embedding_pooling_type=embedding_pooling_type,
            positional_embedding=positional_embedding
        )

        if linked_projection_layer == 0:
            projection_layer = CategoryAndItemProjectionLayer(transformer_hidden_size, len(item_tokenizer),
                                                              len(attribute_tokenizers.get(
                                                                  TOKENIZERS_PREFIX + "." + loss_category)))
        if linked_projection_layer == 1:
            projection_layer = LinkedCategoryAndItemProjectionLayer(linked_projection_layer,transformer_hidden_size, len(item_tokenizer),
                                                                    len(attribute_tokenizers.get(
                                                                        TOKENIZERS_PREFIX + "." + loss_category)))

        if linked_projection_layer in [3,4]:
            projection_layer = ReuseLinkedCategoryAndItemProjectionLayer(linked_projection_layer,transformer_hidden_size, len(item_tokenizer),
                                                                len(attribute_tokenizers.get(
                                                                    TOKENIZERS_PREFIX + "." + loss_category)))

        if linked_projection_layer in [5]:
            projection_layer = CategoryFromItemProjectionLayer(linked_projection_layer,transformer_hidden_size, len(item_tokenizer),
                                                                     len(attribute_tokenizers.get(
                                                                         TOKENIZERS_PREFIX + "." + loss_category)))

        if linked_projection_layer in [6]:
            projection_layer = GoldCategoryAndItemProjectionLayer(linked_projection_layer,transformer_hidden_size, len(item_tokenizer),
                                                               attribute_tokenizers.get(TOKENIZERS_PREFIX + "." + loss_category),
                                                                  loss_category+".target")




        element_representation = NonItemSequenceElementsRepresentationComponent(embedding_layer,
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

    def _add_keys_to_metadata(self, dictionary, metadata_keys):
        if dictionary is not None:
            if dictionary.get(prefusion):
                metadata_keys.extend(list(dictionary[prefusion].keys()))
            if dictionary.get(postfusion):
                metadata_keys.extend(list(dictionary[postfusion].keys()))
            if dictionary.get(sequence_prepend):
                metadata_keys.extend(list(dictionary[sequence_prepend].keys()))

class LinkedCategoryAndItemProjectionLayer(ProjectionLayer):

    @save_hyperparameters
    def __init__(self,
                 linked_layers: int,
                 hidden_size: int,
                 item_vocab_size: int,
                 category_vocab_size: int):
        super().__init__()

        self.item_linear = nn.Linear(hidden_size, item_vocab_size)
        self.category_linear = nn.Linear(hidden_size, category_vocab_size)
        self.mapping_item = nn.Linear(item_vocab_size+category_vocab_size, item_vocab_size)
        self.linked_layers = linked_layers
        if linked_layers == 2:
            self.additional_layer = nn.Linear(item_vocab_size+category_vocab_size,item_vocab_size+category_vocab_size)

    def forward(self,
                modified_sequence_representation: ModifiedSequenceRepresentation
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        representation = modified_sequence_representation.modified_encoded_sequence
        cat_rep = self.category_linear(representation)
        item_scale = self.item_linear(representation)
        concated_cat_item = torch.cat((item_scale,cat_rep), dim=2)
        if self.linked_layers == 1:
            item_rep = self.mapping_item(concated_cat_item)
        else:
            concated_cat_item = self.additional_layer(concated_cat_item)
            item_rep = self.mapping_item(concated_cat_item)
        return item_rep, cat_rep



class ReuseLinkedCategoryAndItemProjectionLayer(ProjectionLayer):

    @save_hyperparameters
    def __init__(self,
                 linked_layers: int,
                 hidden_size: int,
                 item_vocab_size: int,
                 category_vocab_size: int):
        super().__init__()

        self.item_linear = nn.Linear(hidden_size, item_vocab_size)
        self.category_linear = nn.Linear(hidden_size, category_vocab_size)
        self.mapping_item = nn.Linear(hidden_size+category_vocab_size, item_vocab_size)
        self.linked_layers = linked_layers
        if linked_layers == 4:
            self.intermediate = nn.Linear(hidden_size+category_vocab_size,hidden_size+category_vocab_size)

    def forward(self,
                modified_sequence_representation: ModifiedSequenceRepresentation
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        representation = modified_sequence_representation.modified_encoded_sequence
        cat_rep = self.category_linear(representation)
        concated_cat_item = torch.cat((representation, cat_rep), dim=2)
        if self.linked_layers == 4:
            concated_cat_item = self.intermediate(concated_cat_item)
        item_rep = self.mapping_item(concated_cat_item)
        return item_rep, cat_rep


class CategoryFromItemProjectionLayer(ProjectionLayer):

    @save_hyperparameters
    def __init__(self,
                 linked_layers: int,
                 hidden_size: int,
                 item_vocab_size: int,
                 category_vocab_size: int):
        super().__init__()

        self.item_linear = nn.Linear(hidden_size, item_vocab_size)
        self.category_linear = nn.Linear(item_vocab_size, category_vocab_size)

    def forward(self,
                modified_sequence_representation: ModifiedSequenceRepresentation
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        representation = modified_sequence_representation.modified_encoded_sequence
        item_rep = self.item_linear(representation)
        cat_rep = self.category_linear(item_rep)
        return item_rep, cat_rep

class GoldCategoryAndItemProjectionLayer(ProjectionLayer):

    @save_hyperparameters
    def __init__(self,
                 linked_layers: int,
                 hidden_size: int,
                 item_vocab_size: int,
                 cat_tokenizer, gold_cat):
        super().__init__()

        self.category_vocab_size = len(cat_tokenizer)
        self.cat_tokenizer = cat_tokenizer
        self.category_linear = nn.Linear(hidden_size, self.category_vocab_size)
        self.mapping_item = nn.Linear(hidden_size+self.category_vocab_size, item_vocab_size)
        self.linked_layers = linked_layers
        self.gold_cat = gold_cat

    def forward(self,
                modified_sequence_representation: ModifiedSequenceRepresentation
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        representation = modified_sequence_representation.modified_encoded_sequence
        cat_rep = self.category_linear(representation)

        gold_cat = modified_sequence_representation.input_sequence.get_attribute(self.gold_cat)

        #Only at training time
        choice_gold = random_uniform()
        if len(gold_cat.size()) > 2 and (choice_gold > 0.5):
            gold_cat = convert_target_to_multi_hot(gold_cat, len(self.cat_tokenizer), self.cat_tokenizer.pad_token_id)
            concated_cat_item = torch.cat((representation, gold_cat), dim=2)
        else:
            concated_cat_item = torch.cat((representation, cat_rep), dim=2)
        item_rep = self.mapping_item(concated_cat_item)
        return item_rep, cat_rep

class CategoryAndItemProjectionLayer(ProjectionLayer):

    @save_hyperparameters
    def __init__(self,
                 hidden_size: int,
                 item_vocab_size: int,
                 category_vocab_size: int):
        super().__init__()

        self.item_linear = nn.Linear(hidden_size, item_vocab_size)
        self.category_linear = nn.Linear(hidden_size, category_vocab_size)

    def forward(self,
                modified_sequence_representation: ModifiedSequenceRepresentation
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        representation = modified_sequence_representation.modified_encoded_sequence
        cat_rep = self.category_linear(representation)
        item_rep = self.item_linear(representation)
        return item_rep, cat_rep

class CategoryAndItemReuseProjectionLayer(ProjectionLayer):

    @save_hyperparameters
    def __init__(self,
                 item_embedding: nn.Embedding,
                 category_embedding: nn.Embedding,
                 item_vocab_size: int,
                 category_vocab_size: int):
        super().__init__()

        self.item_embedding = item_embedding
        self.category_embedding = category_embedding
        self.item_vocab_size = item_vocab_size
        self.category_vocab_size = category_vocab_size
        self.output_bias_item = nn.Parameter(torch.Tensor(item_vocab_size))
        self.output_bias_category = nn.Parameter(torch.Tensor(category_vocab_size))

        self.init_weights()

    def init_weights(self):
        item_bound = 1 / math.sqrt(self.item_vocab_size)
        nn.init.uniform_(self.output_bias_item, -item_bound, item_bound)
        cat_bound = 1 / math.sqrt(self.category_vocab_size)
        nn.init.uniform_(self.output_bias_category, -cat_bound, cat_bound)

    def forward(self,
                modified_sequence_representation: ModifiedSequenceRepresentation
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        representation = modified_sequence_representation.modified_encoded_sequence
        item_rep = torch.matmul(representation, self.item_embedding.weight.transpose(0, 1))  # (S, N, I)
        item_rep = item_rep + self.output_bias_item
        cat_rep = torch.matmul(representation, self.cat_embedding.weight.transpose(0, 1))  # (S, N, I)
        cat_rep = cat_rep + self.output_bias_category

        return item_rep, cat_rep



def get_attributes(attributes_dictionary, type):
    if attributes_dictionary is not None:
        return attributes_dictionary.get(type, None)
    else:
        return None
