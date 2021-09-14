from typing import List

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories.metrics.metrics import MetricsFactory
from asme.core.init.factories.util import require_config_keys
from asme.core.init.object_factory import ObjectFactory, CanBuildResult
from asme.core.metrics.container.metrics_container import RankingMetricsContainer
from asme.core.metrics.container.metrics_sampler import AllItemsSampler


class FullMetricsFactory(ObjectFactory):
    KEY = 'full'

    def __init__(self):
        super().__init__()
        self.metrics_factory = MetricsFactory()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return require_config_keys(config, ['metrics']) and self.metrics_factory.can_build(config, context)

    def build(self, config: Config, context: Context) -> RankingMetricsContainer:
        metrics = self.metrics_factory.build(config.get_config(self.metrics_factory.config_path()), context)
        return RankingMetricsContainer(metrics, AllItemsSampler())

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
