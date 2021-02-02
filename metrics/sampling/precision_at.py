import torch

from metrics.sampling.common import calc_precision
from metrics.sampling.sampling_metric import SamplingMetric


class PrecisionAtNegativeSamples(SamplingMetric):

    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self._k = k

        self.add_state("precision", torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('count', torch.tensor(0.), dist_reduce_fx="sum")

    def _update(self,
                predictions: torch.Tensor,
                positive_item_mask: torch.Tensor
                ) -> None:
        precision = calc_precision(predictions, positive_item_mask, self._k)

        self.precision += precision.sum()
        self.count += predictions.size()[0]

    def compute(self):
        return self.precision / self.count

    def name(self):
        return f"precision_at_{self._k}/sampled"
