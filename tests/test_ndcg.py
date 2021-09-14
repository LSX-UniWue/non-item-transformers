from typing import Tuple

import pytest
import torch
from util_test_metric import build_sample, EPSILON

from asme.core.metrics.ndcg import NormalizedDiscountedCumulativeGainMetric


def get_single_item_recommendation_samples():
    return [
        build_sample(torch.tensor([[5, 4, 3, 2, 1]]), torch.tensor([[1, 0, 0, 0, 0]]), 1, torch.tensor(1.)),
        build_sample(torch.tensor([[4, 5, 3, 2, 1]]), torch.tensor([[1, 0, 0, 0, 0]]), 1, torch.tensor(0.)),
        build_sample(torch.tensor([[5, 4, 3, 2, 1]]), torch.tensor([[1, 0, 0, 0, 0]]), 3, torch.tensor(1.)),
        build_sample(torch.tensor([[4, 5, 3, 2, 1]]), torch.tensor([[1, 0, 0, 0, 0]]), 3, torch.tensor(0.6309)),
        build_sample(torch.tensor([[3, 4, 5, 2, 1]]), torch.tensor([[1, 0, 0, 0, 0]]), 3, torch.tensor(0.5)),
        build_sample(torch.tensor([[5, 4, 3, 2, 1]]), torch.tensor([[1, 0, 0, 0, 0]]), 5, torch.tensor(1.)),
        build_sample(torch.tensor([[4, 5, 3, 2, 1]]), torch.tensor([[1, 0, 0, 0, 0]]), 5, torch.tensor(0.6309)),
        build_sample(torch.tensor([[3, 4, 5, 2, 1]]), torch.tensor([[1, 0, 0, 0, 0]]), 5, torch.tensor(0.5)),
        build_sample(torch.tensor([[2, 3, 4, 5, 1]]), torch.tensor([[1, 0, 0, 0, 0]]), 5, torch.tensor(0.4306)),
        build_sample(torch.tensor([[1, 2, 3, 4, 5]]), torch.tensor([[1, 0, 0, 0, 0]]), 5, torch.tensor(0.3868)),
    ]


def get_multiple_item_recommendation_samples():
    return [
        build_sample(torch.tensor([[0, 0, 0, 4, 5]]), torch.tensor([[0, 0, 0, 1, 1]]), 2, torch.tensor(1.)),
        build_sample(torch.tensor([[0, 0, 4, 5, 0]]), torch.tensor([[0, 0, 0, 1, 1]]), 2, torch.tensor(1. / (1. + 0.6309))),
        build_sample(torch.tensor([[0, 4, 5, 0, 0]]), torch.tensor([[0, 0, 0, 1, 1]]), 2, torch.tensor(0.)),
        build_sample(torch.tensor([[0, 0, 0, 4, 5]]), torch.tensor([[0, 0, 0, 1, 1]]), 2, torch.tensor(1.)),
        build_sample(torch.tensor([[0, 0, 0, 4, 5]]), torch.tensor([[0, 0, 0, 1, 0]]), 2, torch.tensor(0.6309)),
        build_sample(torch.tensor([[0, 3, 4, 0, 5]]), torch.tensor([[0, 0, 1, 1, 1]]), 2, torch.tensor(1.)),
        build_sample(torch.tensor([[0, 3, 4, 0, 5]]), torch.tensor([[0, 0, 1, 1, 1]]), 3, torch.tensor((1. + 0.6309) / (1. + 0.6309 + 0.5))),
        build_sample(torch.tensor([[0, 3, 4, 0, 5]]), torch.tensor([[0, 0, 1, 1, 1]]), 2, torch.tensor(1.)),
        build_sample(torch.tensor([[0, 3, 4, 0, 5]]), torch.tensor([[1, 0, 0, 0, 1]]), 2, torch.tensor(1. / (1. + 0.6309))),
    ]


class TestNDCG:

    @pytest.mark.parametrize("sample", get_single_item_recommendation_samples())
    def test_single_item_recommendation(self, sample: Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]):
        predictions, target, k, value = sample
        instance = NormalizedDiscountedCumulativeGainMetric(k=k)
        instance.update(predictions, target)

        assert torch.sum(torch.abs(instance.compute() - value)) < EPSILON

    @pytest.mark.parametrize("sample", get_multiple_item_recommendation_samples())
    def test_multiple_item_recommendation(self, sample: Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]):
        predictions, target, k, value = sample
        instance = NormalizedDiscountedCumulativeGainMetric(k=k)
        instance.update(predictions, target)

        assert torch.sum(torch.abs(instance.compute() - value)) < EPSILON
