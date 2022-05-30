import torch
import torch.nn as nn
from asme.core.losses.losses import SequenceRecommenderContrastiveLoss, DEFAULT_REDUCTION, SequenceRecommenderLoss
from torchmetrics.utilities import reduce

from asme.core.tokenization.tokenizer import Tokenizer


class SASRecFullSequenceCrossEntropyLoss(SequenceRecommenderLoss):

    def __init__(self, item_tokenizer: Tokenizer):
        super().__init__()
        self.item_tokenizer = item_tokenizer

    def forward(self, target: torch.Tensor, logit: torch.Tensor) -> torch.Tensor:
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.item_tokenizer.pad_token_id)
        logit_size = logit.size()
        target_size = target.size()

        # we need to accomodate both training with the full target sequence and validation with single targets
        # TODO (AD) determine where to move this logic, that should not be decided by the loss
        if isinstance(target_size, torch.Size) and len(target_size) > 1:
            # we have a full target sequence
            # logit has size: N, S, I needs to be: N*S,I
            logit = torch.reshape(logit, [-1, logit_size[2]])

            # target has size: N, S needs to be: N*S
            target = torch.reshape(target, [-1])

            return loss_fn(logit, target)
        else:
            return loss_fn(logit, target)


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
