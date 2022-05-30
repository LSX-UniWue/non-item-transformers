from typing import List, Any

from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.factories.common.dependencies_factory import DependenciesFactory
from asme.core.init.factories.metrics.topn_metric import TopNMetricFactory
from asme.core.init.factories.metrics.metric import MetricFactory
from asme.core.init.factories.util import can_build_with_subsection, build_with_subsection
from asme.core.init.object_factory import ObjectFactory, CanBuildResult
from asme.core.metrics.dcg import DiscountedCumulativeGainMetric
from asme.core.metrics.f1 import F1Metric
from asme.core.metrics.mrr import MRRMetric
from asme.core.metrics.mrr_full import MRRFullMetric
from asme.core.metrics.ndcg import NormalizedDiscountedCumulativeGainMetric
from asme.core.metrics.precision import PrecisionMetric
from asme.core.metrics.metric import RankingMetric
from asme.core.metrics.rank import Rank
from asme.core.metrics.recall import RecallMetric


def _collect(metrics_dict: [str, List[Any]]) -> List[Any]:
    return [item for sublist in metrics_dict.values() for item in sublist]


class MetricsFactory(ObjectFactory):

    def __init__(self):
        super().__init__()

        # TODO: config this
        self.metrics = DependenciesFactory([TopNMetricFactory('mrr', MRRMetric),
                                            TopNMetricFactory('f1', F1Metric),
                                            TopNMetricFactory('recall', RecallMetric),
                                            TopNMetricFactory('dcg', DiscountedCumulativeGainMetric),
                                            TopNMetricFactory('ndcg', NormalizedDiscountedCumulativeGainMetric),
                                            TopNMetricFactory('precision', PrecisionMetric),
                                            MetricFactory('mrr_full', MRRFullMetric),
                                            MetricFactory('rank', Rank)],
                                           optional_based_on_path=True)

    def can_build(self,  build_context: BuildContext) -> CanBuildResult:
        return can_build_with_subsection(self.metrics, build_context)

    def build(self,  build_context: BuildContext) -> List[RankingMetric]:
        metrics = build_with_subsection(self.metrics, build_context)
        return _collect(metrics)

    def is_required(self, build_context: BuildContext) -> bool:
        pass

    def config_path(self) -> List[str]:
        return ['metrics']

    def config_key(self) -> str:
        return 'metrics'
