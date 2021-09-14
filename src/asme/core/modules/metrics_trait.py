from abc import abstractmethod
from typing import Dict, List, Any

import pytorch_lightning as pl
import torch

from asme.core.metrics.container.metrics_container import MetricsContainer
from asme.core.modules.constants import RETURN_KEY_SEQUENCE, RETURN_KEY_PREDICTIONS, RETURN_KEY_TARGETS, RETURN_KEY_MASK


class MetricsTrait(pl.LightningModule):

    """
        TODO: add documentation
    """

    @abstractmethod
    def get_metrics(self) -> MetricsContainer:
        pass

    def _eval_step_end(self, outputs: Dict[str, torch.Tensor]):
        # (AD) Computation of metrics is moved to *_step_end because some modes, e.g. dp will incorrectly
        # compute the metrics otherwise.
        input_seq = outputs[RETURN_KEY_SEQUENCE]
        prediction = outputs[RETURN_KEY_PREDICTIONS]
        targets = outputs[RETURN_KEY_TARGETS]
        if RETURN_KEY_MASK in outputs:
            mask = outputs[RETURN_KEY_MASK]
        else:
            mask = None

        metrics = self.get_metrics()
        metrics_step_values = metrics.update(input_seq, targets, prediction, mask=mask)
        for name, step_value in metrics_step_values.items():
            self.log(f"{name}", step_value, prog_bar=True)

        return {**metrics_step_values}

    def _eval_epoch_end(self, outputs: List[Any]):
        for name, value in self.get_metrics().compute().items():
            self.log(f"{name}", value, prog_bar=True)

        self.get_metrics().reset()

    def test_step_end(self, outputs: Dict[str, torch.Tensor]):
        return self._eval_step_end(outputs)

    def test_epoch_end(self, outputs: List[Any]):
        self._eval_epoch_end(outputs)

    def validation_step_end(self,
                            outputs: Dict[str, torch.Tensor]):
        return self._eval_step_end(outputs)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self._eval_epoch_end(outputs)
