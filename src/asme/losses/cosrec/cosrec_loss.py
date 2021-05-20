import torch
from asme.losses.losses import RecommenderSequenceContrastiveLoss


class CosRecLoss(RecommenderSequenceContrastiveLoss):

    def forward(self,
                pos_input: torch.Tensor,
                neg_input: torch.Tensor,
                mask: torch.Tensor):
        positive_loss = torch.mean(torch.log(torch.sigmoid(pos_input)))
        negative_loss = torch.mean(torch.log(1 - torch.sigmoid(neg_input)))
        return - positive_loss - negative_loss
