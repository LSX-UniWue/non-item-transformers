from abc import abstractmethod

import torch
from asme.modules.util.module_util import convert_target_to_multi_hot
from torch import nn


DEFAULT_REDUCTION = 'elementwise_mean'


class SequenceRecommenderLoss(nn.Module):

    def __init__(self, reduction: str = DEFAULT_REDUCTION):
        super().__init__()
        self.reduction = reduction

    @abstractmethod
    def forward(self,
                target: torch.Tensor,
                logit: torch.Tensor
                ) -> torch.Tensor:
        """
        calculates the loss of the given positive and negative logits
        :param target: the positive targets :math `(N)` or `(N, BS)`
        :param logit: the logits :math `(N, I)`
        true if the position i should be considered for the loss
        :return: the loss

        where N is batch size and S the max sequence length
        """
        pass


class CrossEntropyLoss(SequenceRecommenderLoss):

    def forward(self,
                target: torch.Tensor,
                logit: torch.Tensor
                ) -> torch.Tensor:
        if len(target.size()) == 1:
            # only one item per sequence step
            loss_fnc = nn.CrossEntropyLoss()
            return loss_fnc(logit, target)

        loss_fnc = nn.BCEWithLogitsLoss()
        target_tensor = convert_target_to_multi_hot(target, len(self.item_tokenizer), self.item_tokenizer.pad_token_id)
        return loss_fnc(logit, target_tensor)


class SequenceRecommenderContrastiveLoss(nn.Module):

    """ A common interface for contrastive losses that use the difference of positive
    and negative samples (e.g. BPR) """

    def __init__(self, reduction: str = DEFAULT_REDUCTION):
        super().__init__()
        self.reduction = reduction

    @abstractmethod
    def forward(self,
                positive_logits: torch.Tensor,
                negative_logits: torch.Tensor,
                mask: torch.Tensor
                ) -> torch.Tensor:
        """
        calculates the loss of the given positive and negative logits
        :param positive_logits: the positive logits :math `(N, S)`
        :param negative_logits: the negative logits :math `(N, S)`
        :param mask: the mask to apply (e.g. for padding) :math `(N, S)`
        true if the position i should be considered for the loss
        :return: the loss

        where N is batch size and S the max sequence length
        """
        pass


class SequenceRecommenderContrastiveItemLoss(nn.Module):

    """ A common interface for contrastive losses that use the difference of positive
    and negative samples but leverage directly the items and logits of all items

     FIXME: merge with the other loss
     """

    def __init__(self, reduction: str = DEFAULT_REDUCTION):
        super().__init__()
        self.reduction = reduction

    @abstractmethod
    def forward(self,
                logits: torch.Tensor,
                positive_items: torch.Tensor,
                negative_items: torch.Tensor,
                ) -> torch.Tensor:
        """
        calculates the loss of the given positive and negative logits
        :param logits: the logits for each item :math `(N, I)`
        :param positive_items: the positive items :math `(N, PI)`
        :param negative_items: the negative items :math `(N, NI)`
        true if the position i should be considered for the loss
        :return: the loss

        where N is batch size and S the max sequence length
        """
        pass
