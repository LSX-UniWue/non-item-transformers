import torch

import torch.nn.functional as F
from typing import Dict, List, Tuple, Callable, Any, Iterator
from pathlib import Path

from torch.utils.data import Sampler, DataLoader
from torch.utils.data.dataset import T_co, Dataset

from asme.core.metrics.metric import MetricStorageMode
from asme.core.modules.metrics_trait import MetricsTrait
import numpy as np
from asme.core.utils.ioutils import load_filtered_vocabulary


def get_positive_item_mask(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Create a positive item mask from the target tensor.

    :param targets: a target tensor (N)
    :param num_classes: the toal number of classes (items)
    :return: a representation where each relevant item is marked with `1` all others are marked with `0`. (N, I)
    """
    return F.one_hot(targets, num_classes)


def _extract_target_indices(input_seq: torch.Tensor, padding_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Finds the model output for the last input item in each sequence.

    :param input_seq: the input sequence. [BS x S]

    :return: the indices for the last input item of the sequence. [BS x I]
    """
    # calculate the padding mask where each non-padding token has the value `1`
    padding_mask = torch.where(input_seq == padding_token_id, 0, 1)
    seq_length = padding_mask.sum(dim=-1) - 1  # [BS]

    batch_index = torch.arange(input_seq.size()[0])  # [BS]

    # select only the outputs at the last step of each sequence
    # target_logits = logits[batch_index, seq_length]  # [BS, I]

    return batch_index, seq_length


def _extract_sample_metrics(module: MetricsTrait) -> List[Tuple[str, torch.Tensor]]:
    """
    Extracts the raw values of all metrics with per-sample-storage enabled.
    :param module: The module used for generating predictions.
    :return: A list of all metrics in the module's metric container with per-sample-storage enabled.
    """
    metrics_container = module.metrics
    metric_names_and_values = list(filter(lambda x: x[1]._storage_mode == MetricStorageMode.PER_SAMPLE,
                                          zip(metrics_container.get_metric_names(),
                                              metrics_container.get_metrics())))
    return list(map(lambda x: (x[0], x[1]), metric_names_and_values))


def _selected_file_and_filter(selected_items_file: Path) -> Tuple[List[int], Callable[[List[int]], Any]]:
    selected_items = None
    if selected_items_file is not None:
        selected_items = load_filtered_vocabulary(selected_items_file)
        selected_items_tensor = torch.tensor(selected_items, dtype=torch.int32)

        def _selected_items_filter(sample_predictions):
            return torch.index_select(sample_predictions, 1, selected_items_tensor)

        filter_predictions = _selected_items_filter
    else:
        def _noop_filter(sample_predictions: np.ndarray):
            return sample_predictions

        filter_predictions = _noop_filter

    return selected_items, filter_predictions



# XXX: currently the predict method returns all batches at once, this is not RAM efficient
# so we loop through the loader and use only one batch to call the predict method of pytorch lightning
# replace as soon as this is fixed in pytorch lighting
class FixedBatchSampler(Sampler):

    def __init__(self,
                 batch_start: int,
                 batch_size: int):
        super().__init__(None)
        self.batch_start = batch_start
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[T_co]:
        return iter([range(self.batch_start, self.batch_start + self.batch_size)])

    def __len__(self):
        return 1

def create_batch_loader(dataset: Dataset,
                         batch_sampler: Sampler,
                         collate_fn,
                         num_workers: int
                         ) -> DataLoader:
    return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=num_workers)


def move_tensors_to_device(obj, device):
    """
    Moves tensors in dict or lists to device, keeps other data on original device.
    :param obj:
    :param device: cpu or cuda device
    :return:
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_tensors_to_device(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_tensors_to_device(v, device))
        return res
    else:
        return obj