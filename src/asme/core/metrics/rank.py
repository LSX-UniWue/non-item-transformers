import torch
from asme.core.metrics.common import get_ranks
from asme.core.metrics.metric import RankingMetric, MetricStorageMode


class Rank(RankingMetric):

    def __init__(self,
                 dist_sync_on_step: bool = False,
                 storage_mode: MetricStorageMode = MetricStorageMode.SUM):
        super().__init__(metric_id='rank',
                         dist_sync_on_step=dist_sync_on_step,
                         storage_mode=storage_mode)

    def _calc_metric(self,
                     prediction: torch.Tensor,
                     positive_item_mask: torch.Tensor,
                     metric_mask: torch.Tensor
                     ) -> torch.Tensor:
        return get_ranks(prediction, positive_item_mask, metric_mask)

    def name(self):
        return f"rank"
