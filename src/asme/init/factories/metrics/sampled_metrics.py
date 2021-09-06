import os
from pathlib import Path
from typing import List

from asme.init.config import Config
from asme.init.context import Context
from asme.init.factories.metrics.metrics import MetricsFactory
from asme.init.factories.util import require_config_keys, infer_whole_path, infer_base_path
from asme.init.object_factory import ObjectFactory, CanBuildResult
from asme.metrics.container.metrics_sampler import NegativeMetricsSampler
from asme.metrics.container.metrics_container import RankingMetricsContainer
from data import CURRENT_SPLIT_PATH_CONTEXT_KEY, DATASET_PREFIX_CONTEXT_KEY


def _load_weights_file(file_path: Path) -> List[float]:
    with open(file_path) as prob_file:
        return [float(line) for line in prob_file.readlines()]


class SampledMetricsFactory(ObjectFactory):
    """
    a factory to build the sampling metrics container for popularity
    """

    KEY = 'sampled'

    def __init__(self):
        super().__init__()
        self.metrics_factory = MetricsFactory()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return require_config_keys(config, ['metrics', 'num_negative_samples', 'sample_probability_file'])

    def build(self, config: Config, context: Context) -> RankingMetricsContainer:
        metrics = self.metrics_factory.build(config.get_config(self.metrics_factory.config_path()), context)
        sample_size = config.get('num_negative_samples')

        # If the provided path is absolute, just use it. Otherwise prepend the location of the current split
        split_path = context.get(CURRENT_SPLIT_PATH_CONTEXT_KEY)
        infer_base_path(config, "sample_probability_file", split_path)
        weights = _load_weights_file(config.get("sample_probability_file"))

        sampler = NegativeMetricsSampler(weights, sample_size, 'sampled')
        return RankingMetricsContainer(metrics, sampler)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
