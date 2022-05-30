from typing import List

from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.core.metrics.metric import RankingMetric


class TopNMetricFactory(ObjectFactory):
    """
    general class to build a RankingMetric for one or multiple ks (topN metric)
    """

    def __init__(self,
                 key: str,
                 ranking_cls):
        super().__init__()
        self.key = key
        self.ranking_cls = ranking_cls

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, build_context: BuildContext) -> List[RankingMetric]:
        ks = build_context.get_current_config_section().get([])
        if not isinstance(ks, list):
            ks = [ks]
        return [self.ranking_cls(k) for k in ks]

    def is_required(self, build_context: BuildContext) -> bool:
        return False

    def config_path(self) -> List[str]:
        return [self.key]

    def config_key(self) -> str:
        return self.key
