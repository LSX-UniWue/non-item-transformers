import torch.nn as nn

from asme.core.models.common.layers.layers import ProjectionLayer, SequenceRepresentationModifierLayer, \
    SequenceElementsRepresentationLayer
from asme.core.models.common.layers.transformer_layers import TransformerLayer
from asme.core.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.core.models.transformer.sequence_representation import \
    TransformerSequenceRepresentationComponent
from asme.core.utils.hyperparameter_utils import save_hyperparameters


class TransformerEncoderModel(SequenceRecommenderModel):
    """
    Basis for all Transformer-Encoder based models.
    """

    @save_hyperparameters
    def __init__(self,
                 transformer_hidden_size: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 transformer_dropout: float,
                 embedding_layer: SequenceElementsRepresentationLayer,
                 sequence_representation_modifier_layer: SequenceRepresentationModifierLayer,
                 projection_layer: ProjectionLayer,
                 bidirectional: bool = False,
                 transformer_intermediate_size: int = None,
                 transformer_attention_dropout: float = None
                 ):
        """
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

        transformer_layer = TransformerLayer(transformer_hidden_size,
                                             num_transformer_heads,
                                             num_transformer_layers,
                                             transformer_intermediate_size,
                                             transformer_dropout,
                                             attention_dropout=transformer_attention_dropout)

        sequence_representation_layer = TransformerSequenceRepresentationComponent(transformer_layer,
                                                                                   bidirectional=bidirectional)

        super().__init__(sequence_embedding_layer=embedding_layer,
                         sequence_representation_layer=sequence_representation_layer,
                         sequence_representation_modifier_layer=sequence_representation_modifier_layer,
                         projection_layer=projection_layer)

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
