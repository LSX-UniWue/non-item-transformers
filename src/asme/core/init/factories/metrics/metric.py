from typing import List

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.core.metrics.metric import RankingMetric


class MetricFactory(ObjectFactory):
    """
    Factory to construct a parameterless RankingMetric
    """
    def __init__(self,
                 key: str,
                 ranking_cls):
        super().__init__()
        self.key = key
        self.ranking_cls = ranking_cls

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> [RankingMetric]:
        # This returns a list instead of a single object due to compatibility issue with the MetricsFactory
        return [self.ranking_cls()]

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return [self.key]

    def config_key(self) -> str:
        return self.key
