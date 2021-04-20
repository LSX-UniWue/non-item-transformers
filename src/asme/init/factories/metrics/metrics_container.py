from typing import List, Union, Any, Dict

from asme.init.config import Config
from asme.init.context import Context
from asme.init.factories.common.dependencies_factory import DependenciesFactory
from asme.init.factories.metrics.fixed_items_metrics import FixedItemsMetricsFactory
from asme.init.factories.metrics.full_metrics import FullMetricsFactory
from asme.init.factories.metrics.sampled_metrics import SampledMetricsFactory
from asme.init.object_factory import ObjectFactory, CanBuildResult
from asme.metrics.container.metrics_container import AggregateMetricsContainer


class MetricsContainerFactory(ObjectFactory):

    """
    factory to build each metrics container
    """

    def __init__(self, metrics_factories: DependenciesFactory = DependenciesFactory([FullMetricsFactory(),
                                                                                     SampledMetricsFactory(),
                                                                                     FixedItemsMetricsFactory()],
                                                                                    optional_based_on_path=True)):
        super().__init__()
        self.metric_factories = metrics_factories

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return self.metric_factories.can_build(config, context)

    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        metrics_dict = self.metric_factories.build(config, context)
        return AggregateMetricsContainer(list(metrics_dict.values()))

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return ['metrics']

    def config_key(self) -> str:
        return "metrics_containers"