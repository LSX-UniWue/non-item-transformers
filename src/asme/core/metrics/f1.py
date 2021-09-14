import torch

from asme.core.metrics.common import calc_precision, calc_recall
from asme.core.metrics.metric import RankingMetric, MetricStorageMode


class F1Metric(RankingMetric):

    """
    calculates the F1 score at k
    """

    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False,
                 storage_mode: MetricStorageMode = MetricStorageMode.SUM):
        super().__init__(metric_id='f1',
                         dist_sync_on_step=dist_sync_on_step,
                         storage_mode=storage_mode)
        self._k = k

    def _calc_metric(self,
                     predictions: torch.Tensor,
                     positive_item_mask: torch.Tensor,
                     metric_mask: torch.Tensor
                     ) -> torch.Tensor:
        precision = calc_precision(predictions, positive_item_mask, self._k, metric_mask)
        recall = calc_recall(predictions, positive_item_mask, self._k, metric_mask)

        # the f1 is the harmonic mean of recall and precision
        f1 = 2 * recall * precision / (recall + precision)
        f1[torch.isnan(f1)] = 0.0
        return f1

    def name(self):
        return f"F1@{self._k}"
