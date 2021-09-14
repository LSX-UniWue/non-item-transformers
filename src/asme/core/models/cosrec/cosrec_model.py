from typing import List

from asme.core.models.common.components.representation_modifier.user_embedding_concat_modifier import \
    UserEmbeddingConcatModifier
from asme.core.models.common.components.projection.sparse_projection_component import SparseProjectionComponent
from asme.core.models.common.components.representations.sequence_embedding import SequenceElementsEmbeddingComponent
from asme.core.models.common.layers.layers import IdentitySequenceRepresentationModifierLayer
from asme.core.models.cosrec.components import CosRecSequenceRepresentationComponent
from asme.core.models.sequence_recommendation_model import SequenceRecommenderModel

from asme.data.datasets import USER_ENTRY_NAME


class CosRecModel(SequenceRecommenderModel):
    """
    A 2D CNN for sequential Recommendation.
    Based on paper "CosRec: 2D Convolutional Neural Networks for Sequential Recommendation" which can be found at
    https://dl.acm.org/doi/10.1145/3357384.3358113.
    Original code used for this model is available at: https://github.com/zzxslp/CosRec.

    Args:
        user_vocab_size: number of users.
        item_vocab_size: number of items.
        max_seq_length: length of sequence, Markov order.
        embed_dim: dimensions for user and item embeddings. (latent dimension in paper)
        block_num: number of cnn blocks. (convolutional layers??)
        block_dim: the dimensions for each block. len(block_dim)==block_num -> List
        fc_dim: dimension of the first fc layer, mainly for dimension reduction after CNN.
        activation_function: type of activation functions (string) to use for the output fcn
        dropout: dropout ratio.
    """

    def __init__(self,
                 user_vocab_size: int,
                 item_vocab_size: int,
                 max_seq_length: int,
                 embed_dim: int,
                 block_num: int,
                 block_dim: List[int],
                 fc_dim: int,
                 activation_function: str,
                 dropout: float,
                 embedding_pooling_type: str = None
                 ):
        user_present = user_vocab_size != 0

        item_embedding = SequenceElementsEmbeddingComponent(vocabulary_size=item_vocab_size,
                                                            embedding_size=embed_dim,
                                                            pooling_type=embedding_pooling_type)

        seq_rep_layer = CosRecSequenceRepresentationComponent(embed_dim, block_num, block_dim, fc_dim, activation_function,
                                                              dropout)
        if user_present:
            mod_layer = UserEmbeddingConcatModifier(user_vocab_size, embed_dim)
        else:
            mod_layer = IdentitySequenceRepresentationModifierLayer()

        representation_size = fc_dim + embed_dim if user_present else fc_dim

        projection_layer = SparseProjectionComponent(item_vocab_size, representation_size)

        super().__init__(item_embedding, seq_rep_layer, mod_layer, projection_layer)

        # user and item embeddings
        item_embedding.elements_embedding.get_weight().data.normal_(0, 1.0 / embed_dim)

    def optional_metadata_keys(self) -> List[str]:
        return [USER_ENTRY_NAME]
