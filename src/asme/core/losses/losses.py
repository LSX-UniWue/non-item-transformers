from abc import abstractmethod

import torch
from torch import nn
import torch.nn.functional as F

from asme.core.tokenization.tokenizer import Tokenizer
from asme.core.modules.util.module_util import convert_target_to_multi_hot


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


# TODO (AD): Deprecated. Will be removed soon.
# TODO (AD): Replace with an Adapter that can be configured to either use Single- or MultiTarget-CEL losses
class CrossEntropyLoss(SequenceRecommenderLoss):

    """
    XXX: discuss: currently we need the item tokenizer to generate the correct targets for the
    basket recommendation setting
    """

    def __init__(self,
                 item_tokenizer: Tokenizer):
        super().__init__()
        self.item_tokenizer = item_tokenizer

    def forward(self,
                target: torch.Tensor,
                logit: torch.Tensor
                ) -> torch.Tensor:
        if len(target.size()) == 1:
            # only one item per sequence step
            loss_fnc = nn.CrossEntropyLoss(ignore_index=self.item_tokenizer.pad_token_id)
            return loss_fnc(logit, target)

        loss_fnc = nn.BCEWithLogitsLoss()
        target_tensor = convert_target_to_multi_hot(target, len(self.item_tokenizer), self.item_tokenizer.pad_token_id)
        return loss_fnc(logit, target_tensor)


class SingleTargetCrossEntropyLoss(SequenceRecommenderLoss):
    """
    A cross entropy loss that supports both single targets for each sequence and targets for each sequence step.

    The loss detects based on the shapes of `target` and `logits` whether a single target for each sequence is given or
    if a target exists for each step in the sequence and adapts accordingly.
    """
    def __init__(self,
                 item_tokenizer: Tokenizer):
        super().__init__()
        self.item_tokenizer = item_tokenizer

    def forward(self,
                target: torch.Tensor,
                logits: torch.Tensor
                ) -> torch.Tensor:
        """
        Calculates the cross-entropy loss for single targets.

        Supports both single targets per sequence and single targets for each sequence step.

        :param target: the targets. [BS] or [BS, I]
        :param logits: the model outputs. [BS, I] or [BS, S, I]

        :return: the loss.
        """
        target_dims = target.size()
        logits_dims = logits.size()

        # only one target for each sequence
        # targets: [BS]
        # logits:  [BS, I]
        if len(target_dims) == 1 and len(logits_dims) == 2:
            # only one item per sequence step
            return F.cross_entropy(logits, target, ignore_index=self.item_tokenizer.pad_token_id)

        # one target for each step in the sequence
        # targets: [BS, S]
        # logits:  [BS, S, I]

        if len(target_dims) == 2 and len(logits_dims) == 3:
            if logits_dims[1] != target_dims[1]:
                raise Exception(f"Number of sequence elements must be equal for logits and targets. "
                                f"logits: {logits_dims}, targets: {target_dims}")

            shaped_logits = torch.reshape(logits, [-1, logits_dims[2]])  # [BS * S, I]
            shaped_target = torch.reshape(target, [-1])  # [BS * S]
            return F.cross_entropy(shaped_logits, shaped_target, ignore_index=self.item_tokenizer.pad_token_id)
        else:
            raise Exception(f"This loss can not be applied to logits and targets with these dimensions: "
                            f"logits: {logits_dims}, target: {target_dims}")


class MultiTargetCrossEntropyLoss(SequenceRecommenderLoss):
    """
    Applies a binary cross entropy loss if multiple targets are predicted for each sequence.

    In contrast to `SingleTargetCrossEntropyLoss` this loss function does not support targets for each step of the
    sequence!
    """
    def __init__(self,
                 item_tokenizer: Tokenizer):
        super().__init__()
        self.item_tokenizer = item_tokenizer

    def forward(self,
                target: torch.Tensor,
                logits: torch.Tensor
                ) -> torch.Tensor:
        """
        Calculates the binary cross entropy loss for multiple targets.

        :param target: the targets. [BS, K]
        :param logits: the model output. [BS, I]
        :return: the loss value.

        `K <= I` is the maximum number of targets per sequence.
        """
        target_tensor = convert_target_to_multi_hot(target, len(self.item_tokenizer), self.item_tokenizer.pad_token_id)
        return F.binary_cross_entropy_with_logits(logits, target_tensor)


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
