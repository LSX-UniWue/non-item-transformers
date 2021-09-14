import torch

from asme.core.metrics.common import get_ranks
from asme.core.metrics.metric import RankingMetric, MetricStorageMode


class MRRFullMetric(RankingMetric):

    """
    calculates the Mean Reciprocal Rank (MRR)
    """

    def __init__(self,
                 dist_sync_on_step: bool = False,
                 storage_mode: MetricStorageMode = MetricStorageMode.SUM):
        super().__init__(metric_id='mrr_full',
                         dist_sync_on_step=dist_sync_on_step,
                         storage_mode=storage_mode)

    def _calc_metric(self,
                     prediction: torch.Tensor,
                     positive_item_mask: torch.Tensor,
                     metric_mask: torch.Tensor
                     ) -> torch.Tensor:
        rank = get_ranks(prediction, positive_item_mask, metric_mask)

        # mrr will contain 'inf' values if target is not in top k scores -> setting it to 0
        mrr = 1 / rank
        mrr[mrr == float('inf')] = 0
        return mrr

    def name(self):
        return f"MRR"
