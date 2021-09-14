import torch.nn as nn

from asme.core.models.common.layers.layers import IdentitySequenceRepresentationModifierLayer
from asme.core.models.common.layers.transformer_layers import TransformerEmbedding, TransformerLayer
from asme.core.models.sasrec.components import SASRecTransformerComponent, SASRecProjectionComponent
from asme.core.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.core.utils.hyperparameter_utils import save_hyperparameters


class SASRecModel(SequenceRecommenderModel):
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
                 item_vocab_size: int,
                 max_seq_length: int,
                 transformer_dropout: float,
                 embedding_pooling_type: str = None,
                 transformer_intermediate_size: int = None,
                 transformer_attention_dropout: float = None
                 ):
        """
        inits the SASRec model
        :param transformer_hidden_size: the hidden size of the transformer
        :param num_transformer_heads: the number of heads of the transformer
        :param num_transformer_layers: the number of layers of the transformer
        :param item_vocab_size: the item vocab size
        :param max_seq_length: the max sequence length
        :param transformer_dropout: the dropout of the model
        :param embedding_pooling_type: the pooling to use for basket recommendation
        :param transformer_intermediate_size: the intermediate size of the transformer (default 4 * transformer_hidden_size)
        :param transformer_attention_dropout: the attention dropout (default transformer_dropout)
        """

        if transformer_intermediate_size is None:
            transformer_intermediate_size = 4 * transformer_hidden_size

        embedding_layer = TransformerEmbedding(item_voc_size=item_vocab_size,
                                               max_seq_len=max_seq_length,
                                               embedding_size=transformer_hidden_size,
                                               dropout=transformer_dropout,
                                               embedding_pooling_type=embedding_pooling_type)

        transformer_layer = TransformerLayer(transformer_hidden_size,
                                             num_transformer_heads,
                                             num_transformer_layers,
                                             transformer_intermediate_size,
                                             transformer_dropout,
                                             attention_dropout=transformer_attention_dropout)
        sasrec_transformer_layer = SASRecTransformerComponent(transformer_layer)

        modified_seq_representation_layer = IdentitySequenceRepresentationModifierLayer()
        sasrec_projection_layer = SASRecProjectionComponent(embedding_layer)

        super().__init__(sequence_embedding_layer=embedding_layer,
                         sequence_representation_layer=sasrec_transformer_layer,
                         sequence_representation_modifier_layer=modified_seq_representation_layer,
                         projection_layer=sasrec_projection_layer)

        # FIXME (AD) I think we should move this out of the model and call it through a callback before training starts
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
