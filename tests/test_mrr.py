from typing import Tuple

import pytest
import torch
from util_test_metric import build_sample, EPSILON

from asme.core.metrics.mrr import MRRMetric


def get_single_item_recommendation_samples():
    return [
        build_sample(torch.tensor([[5, 4, 3, 2, 1]]), torch.tensor([[1, 0, 0, 0, 0]]), 1, torch.tensor(1.)),
        build_sample(torch.tensor([[4, 5, 3, 2, 1]]), torch.tensor([[1, 0, 0, 0, 0]]), 1, torch.tensor(0.)),
        build_sample(torch.tensor([[5, 4, 3, 2, 1]]), torch.tensor([[1, 0, 0, 0, 0]]), 3, torch.tensor(1.)),
        build_sample(torch.tensor([[4, 5, 3, 2, 1]]), torch.tensor([[1, 0, 0, 0, 0]]), 3, torch.tensor(0.5)),
        build_sample(torch.tensor([[3, 4, 5, 2, 1]]), torch.tensor([[1, 0, 0, 0, 0]]), 3, torch.tensor(0.33333)),
        build_sample(torch.tensor([[2, 3, 4, 5, 1]]), torch.tensor([[1, 0, 0, 0, 0]]), 3, torch.tensor(0.)),
        build_sample(torch.tensor([[5, 4, 3, 2, 1]]), torch.tensor([[1, 0, 0, 0, 0]]), 5, torch.tensor(1.)),
        build_sample(torch.tensor([[4, 5, 3, 2, 1]]), torch.tensor([[1, 0, 0, 0, 0]]), 5, torch.tensor(0.5)),
        build_sample(torch.tensor([[3, 4, 5, 2, 1]]), torch.tensor([[1, 0, 0, 0, 0]]), 5, torch.tensor(0.33333)),
        build_sample(torch.tensor([[2, 3, 4, 5, 1]]), torch.tensor([[1, 0, 0, 0, 0]]), 5, torch.tensor(0.25)),
        build_sample(torch.tensor([[1, 2, 3, 4, 5]]), torch.tensor([[1, 0, 0, 0, 0]]), 5, torch.tensor(0.2)),
    ]


class TestMRR:

    @pytest.mark.parametrize("sample", get_single_item_recommendation_samples())
    def test_single_item_recommendation(self, sample: Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]):
        predictions, target, k, value = sample
        instance = MRRMetric(k=k)
        instance.update(predictions, target)

        assert torch.sum(torch.abs(instance.compute() - value)) < EPSILON
