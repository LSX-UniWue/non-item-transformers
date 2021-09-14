from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor


@dataclass
class MetricsSample:
    sampled_predictions: Tensor
    """
    A tensor containing the predictions of the model for the sampled items.
    """

    positive_item_mask: Tensor
    """
    A mask that marks the positions of the positive target items in the :code sampled_predictions tensor.
    """

    metric_mask: Optional[Tensor]
    """
    A mask that masks the positions that should be considered by the calculation of the loss
    """


class MetricsSampler:
    """
    abstract class to implement to sample only a subset of items for evaluation
    """

    @abstractmethod
    def sample(self,
               input_seq: torch.Tensor,
               targets: torch.Tensor,
               predictions: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> MetricsSample:
        pass

    @abstractmethod
    def suffix_metric_name(self) -> str:
        pass


class AllItemsSampler(MetricsSampler):

    """
        A sampler that samples all items
    """

    def sample(self,
               input_seq: torch.Tensor,
               targets: torch.Tensor,
               predictions: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> MetricsSample:
        positive_item_mask = _to_multi_hot(predictions.size(), targets)
        return MetricsSample(predictions, positive_item_mask, None)

    def suffix_metric_name(self) -> str:
        return ""


def _to_multi_hot(size: List[int],
                  targets: torch.Tensor
                  ) -> torch.Tensor:
    device = targets.device
    multihot = torch.zeros(size).to(torch.long).to(device=device)
    src = torch.ones_like(multihot).to(torch.long)
    if len(targets.size()) == 1:
        targets = targets.unsqueeze(1)
    return multihot.scatter(1, targets, src)


class FixedItemsSampler(MetricsSampler):
    """
    this sampler always returns only the configured items
    """

    def __init__(self, fixed_items: List[int]):
        super().__init__()
        self.fixed_items = fixed_items

    def sample(self,
               input_seq: torch.Tensor,
               targets: torch.Tensor,
               predictions: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> MetricsSample:
        fixed_items = torch.LongTensor(self.fixed_items).to(targets.device)
        fixed_items = fixed_items.unsqueeze(0).repeat(targets.size()[0], 1)

        sampled_predictions = predictions.gather(1, fixed_items)

        if len(targets.size()) == 1:
            # next-item recommendation
            target_batched = targets.unsqueeze(1)
            positive_item_mask = fixed_items.eq(target_batched).to(dtype=predictions.dtype)
        else:
            # here we multi-hot encode the targets (1 for each target in the item space)
            multihot = _to_multi_hot(predictions.size(), targets)
            multihot[:, 0] = 0
            # and than gather the corresponding indices by the fixed items
            positive_item_mask = multihot.gather(1, fixed_items)

        return MetricsSample(sampled_predictions, positive_item_mask, None)

    def suffix_metric_name(self) -> str:
        return "_fixed"


class NegativeMetricsSampler(MetricsSampler):
    """
    Generates negative samples based on the target item space, targets and actual input.

    The sampler draws from the target item distribution as defined by the :code weights provided
    on initialization. Also care is taken that the target items and items in the input sequence do not appear in the
    generated sample.

    After the sample is drawn, the target scores for the target items are added.

    Example:
        If the :code sample_size is :code 100, and there is one target, the final sample has size :code 101
    """

    def __init__(self,
                 weights: List[float],
                 sample_size: int,
                 metrics_suffix: str):
        """
        Constructor.

        :param weights: weights for each item in target space.
        :param sample_size: number of samples generated for evaluation.
        :param metrics_suffix: the suffix of the metrics
        """
        super().__init__()
        self.weights = weights
        self.sample_size = sample_size
        self.metrics_suffix = metrics_suffix

    def sample(self,
               input_seq: torch.Tensor,
               targets: torch.Tensor,
               predictions: torch.Tensor,
               mask: Optional[torch.Tensor] = None
               ) -> MetricsSample:
        """
        Generates :code weights negative samples for each entry in the batch.
        
        :param input_seq: the batch of input sequences. :math (N, S) or (N, S, BS)
        :param targets:  the targets for the provided batch. :math (N) or (N, BS)
        :param predictions: the predictions for the input_seq made by the model. :math (N, I)
        :param mask: the mask of padded target items (only when basket recommendation)
         
        :return: the metrics sample containing the predictions to consider and the positive item mask (where a 1 indices
        that the item was the target)

        where N is the batch size, S the max sequence length, BS the max basket size and I the vocab item size
        """
        batch_size = input_seq.size()[0]
        weight = torch.tensor(self.weights, device=predictions.device)
        weight = weight.unsqueeze(0).repeat(batch_size, 1)

        target_batched = targets

        # never sample targets
        is_basket_recommendation = len(targets.size()) == 2
        if is_basket_recommendation:
            target_src = torch.ones_like(targets).to(torch.long)
            target_mask = torch.zeros_like(weight).to(torch.long)
            target_mask = target_mask.scatter(1, targets, target_src)
            weight[target_mask.to(dtype=torch.bool)] = 0.

            # we flatten the input sequence so there are all items in all baskets in the second dimension
            input_seq = input_seq.view(batch_size, -1)
        else:
            batch_indices = torch.arange(0, batch_size)
            weight[batch_indices, targets] = 0.
            target_batched = targets.unsqueeze(1)

        # we want to use scatter to set the items contained in the sequence to 0 for every row in the batch
        src = torch.ones_like(input_seq).to(torch.long)
        sequence_mask = torch.zeros_like(weight).to(torch.long)

        # calculate a mask where 1. signals that the item should get 0. probability since it occurs in the input
        # sequence.
        sequence_mask = sequence_mask.scatter(1, input_seq, src)
        weight[sequence_mask.to(dtype=torch.bool)] = 0.

        sampled_negatives = torch.multinomial(weight, num_samples=self.sample_size)

        sampled_items = torch.cat([target_batched, sampled_negatives], dim=1)

        metric_mask = torch.ones_like(sampled_items)
        # now generate the positive item mask
        if is_basket_recommendation:
            negative_items_mask = torch.zeros_like(sampled_negatives)
            positive_item_mask = torch.cat([mask, negative_items_mask], dim=1)
            metric_mask = torch.cat([mask, torch.ones_like(sampled_negatives)], dim=1)
        else:
            positive_item_mask = sampled_items.eq(target_batched).to(dtype=predictions.dtype)

        # gather the predictions of all consider items
        sampled_predictions = predictions.gather(1, sampled_items)
        return MetricsSample(sampled_predictions, positive_item_mask, metric_mask)

    def suffix_metric_name(self) -> str:
        return f"_{self.metrics_suffix}({self.sample_size})"
