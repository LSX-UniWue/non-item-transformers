from typing import Optional, Callable

import torch
from torch.optim import Optimizer


class NoopOptimizer(Optimizer):
    """
        Optimizer that does nothing. Useful for counting-based baselines which do not need loss-based training.
    """
    def __init__(self):
        super().__init__([torch.tensor([0.])], {})

    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        closure()
        return None

