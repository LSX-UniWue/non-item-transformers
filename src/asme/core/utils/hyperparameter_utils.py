import inspect
from functools import wraps
from torch import nn
from asme.core.tokenization.tokenizer import Tokenizer
import pytorch_lightning as pl

from asme.core.metrics.container.metrics_container import MetricsContainer


def _get_hyperparameters(args, kwargs, init_func):

    parameters = list(inspect.signature(init_func).parameters.items())

    hyperparameters = {}
    # first the args

    for index, arg in enumerate(args):
        # index + 1 because self is the first parameter of init
        hyperparameters[parameters[index + 1][0]] = arg

    for index, arg in enumerate(kwargs):
        value = kwargs.get(arg, None)
        if isinstance(value, (MetricsContainer, pl.metrics.Metric, Tokenizer)):  # excluding non-hyperparameters like Metrics and Tokenizer
            continue
        hyperparameters[arg] = value
        if isinstance(value, nn.Module) and hasattr(value, 'hyperparameters'):                    # special handling of the model parameter
            model_hyperparameters = value.hyperparameters
            hyperparameters[arg] = model_hyperparameters

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
