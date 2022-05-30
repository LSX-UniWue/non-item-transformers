from pathlib import Path
from typing import List

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.factories.metrics.metrics import MetricsFactory
from asme.core.init.factories.util import require_config_keys, infer_base_path, build_with_subsection
from asme.core.init.object_factory import ObjectFactory, CanBuildResult
from asme.core.metrics.container.metrics_sampler import NegativeMetricsSampler
from asme.core.metrics.container.metrics_container import RankingMetricsContainer
from asme.data import CURRENT_SPLIT_PATH_CONTEXT_KEY


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

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        return require_config_keys(build_context.get_current_config_section(),
                                   ['metrics', 'num_negative_samples', 'sample_probability_file'])

    def build(self, build_context: BuildContext) -> RankingMetricsContainer:
        metrics = build_with_subsection(self.metrics_factory, build_context)
        config = build_context.get_current_config_section()

        sample_size = config.get('num_negative_samples')

        # If the provided path is absolute, just use it. Otherwise prepend the location of the current split
        split_path = build_context.get_context().get(CURRENT_SPLIT_PATH_CONTEXT_KEY)
        infer_base_path(config, "sample_probability_file", split_path)
        weights = _load_weights_file(config.get("sample_probability_file"))

        sampler = NegativeMetricsSampler(weights, sample_size, 'sampled')
        return RankingMetricsContainer(metrics, sampler)

    def is_required(self, build_context: BuildContext) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
