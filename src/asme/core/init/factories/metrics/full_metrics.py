from typing import List

from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.factories.metrics.metrics import MetricsFactory
from asme.core.init.factories.util import require_config_keys, can_build_with_subsection, build_with_subsection
from asme.core.init.object_factory import ObjectFactory, CanBuildResult
from asme.core.metrics.container.metrics_container import RankingMetricsContainer
from asme.core.metrics.container.metrics_sampler import AllItemsSampler


class FullMetricsFactory(ObjectFactory):
    KEY = 'full'

    def __init__(self):
        super().__init__()
        self.metrics_factory = MetricsFactory()

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        return require_config_keys(build_context.get_current_config_section(), ['metrics']) and \
               can_build_with_subsection(self.metrics_factory, build_context)

    def build(self, build_context: BuildContext) -> RankingMetricsContainer:
        metrics = build_with_subsection(self.metrics_factory, build_context)
        return RankingMetricsContainer(metrics, AllItemsSampler())

    def is_required(self, build_context: BuildContext) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
