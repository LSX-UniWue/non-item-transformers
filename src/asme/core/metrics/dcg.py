import torch

from asme.core.metrics.common import calc_dcg
from asme.core.metrics.metric import RankingMetric, MetricStorageMode


class DiscountedCumulativeGainMetric(RankingMetric):
    """
    calculates the Discounted Cumulative Gain (DCG) at k
    """

    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False,
                 storage_mode: MetricStorageMode = MetricStorageMode.SUM):
        super().__init__(metric_id='dcg',
                         dist_sync_on_step=dist_sync_on_step,
                         storage_mode=storage_mode)
        self._k = k

    def _calc_metric(self,
                     prediction: torch.Tensor,
                     positive_item_mask: torch.Tensor,
                     metric_mask: torch.Tensor
                     ) -> torch.Tensor:
        return calc_dcg(prediction, positive_item_mask, self._k, metric_mask)

    def name(self):
        return f"DCG@{self._k}"
