from abc import abstractmethod
from enum import Enum
from typing import Optional, Callable, Any, Union, List

import pytorch_lightning as pl
import torch


class MetricStorageMode(Enum):
    """ enum describing all possible modes the ranking metrics support for storing the results """

    """ stores each result (e.g. used while predicting) """
    PER_SAMPLE = 'per_sample'
    """ stores only a sum (e.g. used while training) """
    SUM = 'sum'


class RankingMetric(pl.metrics.Metric):
    """
    base class to implement metrics in the framework
    """

    def __init__(self,
                 metric_id: str,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,
                 storage_mode: MetricStorageMode = MetricStorageMode.SUM):
        """
        :param storage_mode: If set to True, this metric object will keep track of the metric values of
        every sample as opposed to averaging over all samples it receives (Useful in evaluation settings with small
        subsets of items). Enable this feature with care since it requires substantial amounts of memory.
        The memory requirements rise linearly with every sample seen.
        """
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn)

        self._metric_id = metric_id
        self._storage_mode = None
        self.set_metrics_storage_mode(storage_mode)

    def set_metrics_storage_mode(self,
                                 storage_mode: MetricStorageMode
                                 ):
        self._storage_mode = storage_mode
        if self._storage_mode == MetricStorageMode.PER_SAMPLE:
            self.add_state(self._metric_id, [], dist_reduce_fx="cat")
        else:
            self.add_state(self._metric_id, torch.tensor(0.), dist_reduce_fx="sum")

        self.add_state('count', torch.tensor(0), dist_reduce_fx="sum")

    def update(self,
               predictions: torch.Tensor,
               positive_item_mask: torch.Tensor,
               metric_mask: torch.Tensor = None
               ) -> None:
        """
        :param predictions: the prediction logits :math `(N, I)`
        :param positive_item_mask: the positive item mask 1 if the item at index i is relevant :math `(N)`
        :param metric_mask: a mask to mask single item in the prediction :math `(N)`

        where N is the batch size and I the item size to evaluate (can be the item vocab size)
        """
        if metric_mask is None:
            metric_mask = torch.ones_like(positive_item_mask)

        # calculate the metrics
        new_metrics = self._calc_metric(predictions, positive_item_mask, metric_mask)

        metrics = self.raw_metric_values()

        if self._storage_mode == MetricStorageMode.PER_SAMPLE:
            metrics += [new_metrics]
        else:
            metrics += new_metrics.sum()
        self.count += predictions.size()[0]

    @abstractmethod
    def _calc_metric(self,
                     prediction: torch.Tensor,
                     positive_item_mask: torch.Tensor,
                     metric_mask: torch.Tensor
                     ) -> torch.Tensor:
        pass

    def compute(self):
        metric = self.raw_metric_values()
        if self._storage_mode == MetricStorageMode.PER_SAMPLE:
            return torch.cat(metric).sum() / self.count
        else:
            return metric / self.count

    def raw_metric_values(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        return getattr(self, self._metric_id)

    @abstractmethod
    def name(self) -> str:
        """
        Returns the name that identifies this metric.

        :return: the name.
        """
        pass
