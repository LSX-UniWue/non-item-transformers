import torch
from torch import nn as nn

from asme.core.models.common.layers.sequence_embedding import SequenceElementsEmbeddingLayer


class LocalEncoderLayer(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 latent_size: int
                 ):
        """
        Local encoder as defined in "Neural Attentive Session-based Recommendation." (https://dl.acm.org/doi/10.1145/3132847.3132926).

        :param hidden_size: hidden size of the encoder RNN.
        :param latent_size: size of the latent space used for computing the alphas.
        """
        super(LocalEncoderLayer, self).__init__()
        self.A1 = nn.Linear(hidden_size, latent_size, bias=False)
        self.A2 = nn.Linear(hidden_size, latent_size, bias=False)
        self.v = nn.Parameter(torch.Tensor(latent_size))

        self.projection_activation = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.uniform_(self.v, -1.0, 1.0)

    def forward(self,
                s1: torch.Tensor,
                s2: torch.Tensor,
                mask: torch.Tensor
                ):
        """
        Calculates v * sigmoid(A1 * s1 + A2 * s2)
        :param s1: a tensor (N, H)
        :param s2: a tensor (N, S, H)
        :param mask: a tensor TODO: alex: please add the dimension

        :return: :math `(N, S, 1)`
        """
        s1_prj = self.A1(s1)  # (N, H)
        s2_prj = self.A2(s2)  # (N, S, H)

        # we need to repeat s_1 for every step in the batch to calculate all steps at once
        s1_prj = torch.unsqueeze(s1_prj, dim=1)  # (N, 1, H)
        s1_prj = torch.repeat_interleave(s1_prj, repeats=s2_prj.size()[1], dim=1)  # (N, S, H)
        sum_projection = s1_prj + s2_prj
        state_representation = self.projection_activation(sum_projection)
        alphas = torch.matmul(state_representation, self.v)  # (N, S),  (v: (1, H) * (N, S, H))
        alphas = torch.unsqueeze(alphas, dim=2)  # (N, S, 1) one scalar weight for every position in the sequence

        weighted = alphas * s2
        # (N, S, H): the alphas get broadcasted along the last dimension and then a component-wise multiplication
        # is performed, scaling every step differently

        # (N, S, 1) make sure that the mask value for each sequence member is broadcasted to all features
        # -> zeroing out masked features
        mask = torch.unsqueeze(mask, dim=-1)

        weighted = mask.to(dtype=weighted.dtype) * weighted  # (N, S, H)

        return torch.sum(weighted, dim=1)  # (N, H)


class BilinearDecoderLayer(nn.Module):
    """
        Implementation of the bilinear decoder
    """
    def __init__(self,
                 embedding_layer: SequenceElementsEmbeddingLayer,
                 encoded_representation_size: int,
                 apply_softmax: bool = False
                 ):
        """

        :param embedding_layer: an item embedding layer
        :param encoded_representation_size: the full size of the encoded representation
        :param apply_softmax: whether to apply softmax or not.
        """
        super().__init__()

        self.embedding_layer = embedding_layer
        # TODO: rename
        self.B = nn.Linear(embedding_layer.embedding.weight.size()[1], encoded_representation_size, bias=False)
        self.activation = nn.Softmax() if apply_softmax else nn.Identity()

    def forward(self,
                context: torch.Tensor,
                items: torch.Tensor = None
                ):
        """
        Computes the similarity scores for the context with each item in 'items'.
        If 'items' is none a similarity score for every item will be computed.

        :param context: a context tensor (N, H).
        :param items: a tensor with items (NI).

        :return: the similarity scores (N, NI),

        where N is the batch size and NI the number of items (can be I)
        """
        # context: (N, H)
        # self.B: (E, H)

        if not items:
            items = torch.arange(self.embedding_layer.embedding.weight.size()[0], dtype=torch.long, device=context.device)
        # items: NI <- number of items evaluated

        embi = self.embedding_layer(items, flatten=False)  # (NI, E)
        embi_b = self.B(embi)  # (NI, H)
        embi_b_T = embi_b.T  # (H, NI)

        similarities = torch.matmul(context, embi_b_T)  # (N, NI) <- a score for each item

        return self.activation(similarities)