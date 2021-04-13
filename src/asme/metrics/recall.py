import torch

from asme.metrics.common import calc_recall
from asme.metrics.metric import RankingMetric, MetricStorageMode


class RecallMetric(RankingMetric):

    """
    calculates the recall at k
    """

    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False,
                 storage_mode: MetricStorageMode = MetricStorageMode.SUM):
        super().__init__(metric_id='recall',
                         dist_sync_on_step=dist_sync_on_step,
                         storage_mode=storage_mode)
        self._k = k

    def _calc_metric(self,
                     prediction: torch.Tensor,
                     positive_item_mask: torch.Tensor,
                     metric_mask: torch.Tensor
                     ) -> torch.Tensor:
        return calc_recall(prediction, positive_item_mask, self._k, metric_mask)

    def name(self):
        return f"recall@{self._k}"
