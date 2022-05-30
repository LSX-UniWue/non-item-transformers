from typing import List, Union, Any, Dict

from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.factories.common.dependencies_factory import DependenciesFactory
from asme.core.init.factories.metrics.fixed_items_metrics import FixedItemsMetricsFactory
from asme.core.init.factories.metrics.full_metrics import FullMetricsFactory
from asme.core.init.factories.metrics.random_sampled_metric import RandomSampledMetricsFactory
from asme.core.init.factories.metrics.sampled_metrics import SampledMetricsFactory
from asme.core.init.factories.util import can_build_with_subsection, build_with_subsection
from asme.core.init.object_factory import ObjectFactory, CanBuildResult
from asme.core.metrics.container.metrics_container import AggregateMetricsContainer


class MetricsContainerFactory(ObjectFactory):

    """
    factory to build each metrics container
    """

    def __init__(self, metrics_factories: DependenciesFactory = DependenciesFactory([FullMetricsFactory(),
                                                                                     SampledMetricsFactory(),
                                                                                     RandomSampledMetricsFactory(),
                                                                                     FixedItemsMetricsFactory()],
                                                                                    optional_based_on_path=True)):
        super().__init__()
        self.metric_factories = metrics_factories

    def can_build(self,  build_context: BuildContext) -> CanBuildResult:
        return can_build_with_subsection(self.metric_factories, build_context)

    def build(self,  build_context: BuildContext) -> Union[Any, Dict[str, Any], List[Any]]:
        metrics_dict = build_with_subsection(self.metric_factories, build_context)
        return AggregateMetricsContainer(list(metrics_dict.values()))

    def is_required(self, build_context: BuildContext) -> bool:
        return True

    def config_path(self) -> List[str]:
        return ['metrics']

    def config_key(self) -> str:
        return "metrics_containers"
