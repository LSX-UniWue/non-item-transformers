import torch

from asme.core.metrics.common import get_true_positives
from asme.core.metrics.metric import RankingMetric


class MRRMetric(RankingMetric):

    """
    calculates the Mean Reciprocal Rank (MRR) at k
    """
    def __init__(self,
                 k: int,
                 storage_mode: bool = False,
                 dist_sync_on_step: bool = True):
        super().__init__(metric_id='mrr',
                         dist_sync_on_step=dist_sync_on_step,
                         storage_mode=storage_mode)
        self._k = k

    def _calc_metric(self,
                     prediction: torch.Tensor,
                     positive_item_mask: torch.Tensor,
                     metric_mask: torch.Tensor
                     ) -> torch.Tensor:
        device = prediction.device

        tp = get_true_positives(prediction, positive_item_mask, self._k, metric_mask)

        num_positions = min(prediction.size()[1], self._k) + 1
        ranks = torch.arange(1, num_positions).unsqueeze(0).repeat(prediction.size()[0], 1).to(device=device)
        rank = (ranks * tp).max(dim=-1).values

        # mrr will contain 'inf' values if target is not in top k scores -> setting it to 0
        mrr = 1 / rank
        mrr[mrr == float('inf')] = 0
        return mrr

    def name(self):
        return f"MRR@{self._k}"
