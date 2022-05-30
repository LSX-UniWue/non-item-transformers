import inspect
from functools import wraps
from typing import Dict, Any

import torchmetrics.metric
from torch import nn
from asme.core.tokenization.tokenizer import Tokenizer
import pytorch_lightning as pl

from asme.core.metrics.container.metrics_container import MetricsContainer
from asme.core.utils.logging import get_logger

logger = get_logger(__name__)


def _get_module_hyperparameters(param: str, module: nn.Module) -> Dict[str, Any]:
    if hasattr(module, 'hyperparameters'):  # special handling of the model parameter
        return module.hyperparameters
    else:
        logger.warning(f"Your module contains a submodule named <{param}>"
                       f" that does not report its hyperparameters,"
                       f" consider adding @save_hyperparameters annotation.")
        return {}


def _get_hyperparameters(args, kwargs, init_func):

    parameters = list(inspect.signature(init_func).parameters.items())

    hyperparameters = {}
    params = kwargs.copy()
    for index, arg in enumerate(args):
        params[parameters[index + 1][0]] = arg

    for index, arg in enumerate(params):
        value = params.get(arg, None)
        # Exclude non-hyperparameters such as MetricContainers, Metrics, etc.
        if isinstance(value, (MetricsContainer, torchmetrics.metric.Metric, Tokenizer, type)):
            continue
        # Check all elements of a list separately
        elif isinstance(value, list):
            hyperparameters[arg] = []
            for i, entry in enumerate(value):
                if isinstance(entry, nn.Module):
                    hyperparameters[arg] += [_get_module_hyperparameters(f"<{arg}>.{i}", entry)]
                else:
                    hyperparameters[arg] = value
        # Check all entries of a dict separately
        elif isinstance(value, dict):
            hyperparameters[arg] = {}
            for key, entry in value.items():
                if isinstance(value, nn.Module):
                    hyperparameters[arg][key] = _get_module_hyperparameters(f"<{arg}>.{key}", entry)
                else:
                    hyperparameters[arg][key] = value
        # If the parameter is a Module, try to retrieve its hyperparameters
        elif isinstance(value, nn.Module):
            hyperparameters[arg] = _get_module_hyperparameters(arg, value)
        # If its just some object or primitive, copy the value.
        else:
            hyperparameters[arg] = value

    return hyperparameters


def save_hyperparameters(init):
    """
    overrides init to set the init values as hyperparameters of the model
    FIXME: currently the module must call self.save_hyperparameters(self.hyperparameters) by itself
    :param init:
    :return:
    """

    @wraps(init)
    def new_init(self, *args, **kwargs):
        self.hyperparameters = _get_hyperparameters(args, kwargs, init)
        init(self, *args, **kwargs)

    return new_init
