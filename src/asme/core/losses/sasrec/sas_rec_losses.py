import torch
from asme.core.losses.losses import SequenceRecommenderContrastiveLoss, DEFAULT_REDUCTION
from pytorch_lightning.metrics.utils import reduce


class SASRecBinaryCrossEntropyLoss(SequenceRecommenderContrastiveLoss):
    """
    The adapted binary cross entropy loss from the SASRec paper
    """

    def forward(self,
                pos_input: torch.Tensor,
                neg_input: torch.Tensor,
                mask: torch.Tensor):
        return sas_rec_binary_cross_entropy(pos_input, neg_input, mask, self.reduction)


def _log_sigmoid(tensor: torch.Tensor,
                 eps: float = 1e-24,
                 reverse: bool = False) -> torch.Tensor:
    tensor_eps = torch.sigmoid(tensor)
    if reverse:
        tensor_eps = 1 - tensor_eps
    return torch.log(tensor_eps + eps)


def sas_rec_binary_cross_entropy(pos_input: torch.Tensor,
                                 neg_input: torch.Tensor,
                                 mask: torch.Tensor,
                                 reduction: str = DEFAULT_REDUCTION
                                 ) -> torch.Tensor:
    """
    :param pos_input: the positive logits :math `(N, S)`
    :param neg_input: the negative logits :math `(N, S)`
    :param mask: the mask to apply (padding) :math `(N, S)`
    :param reduction: the reduction to perform
    :return: the SASRec binary corss entropy for the given inputs

    where N is batch size and S the max sequence length
    """

    pos = _log_sigmoid(pos_input) * mask
    neg = _log_sigmoid(neg_input, reverse=True) * mask

    avg_sum = torch.sum(- pos - neg) / torch.sum(mask)
    return reduce(avg_sum, reduction)
