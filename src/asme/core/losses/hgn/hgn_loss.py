import torch
from asme.core.losses.losses import SequenceRecommenderContrastiveLoss


class HGNLoss(SequenceRecommenderContrastiveLoss):

    def forward(self,
                positive_logits: torch.Tensor,
                negative_logits: torch.Tensor,
                mask: torch.Tensor
                ) -> torch.Tensor:
        loss = - torch.log(torch.sigmoid(positive_logits - negative_logits) + 1e-8)
        return torch.mean(torch.sum(loss))
