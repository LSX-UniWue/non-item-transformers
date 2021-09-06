import os.path
from pathlib import Path
from typing import List

from asme.init.config import Config
from asme.init.context import Context
from asme.init.factories.metrics.metrics import MetricsFactory
from asme.init.factories.util import require_config_keys, infer_base_path
from asme.init.object_factory import ObjectFactory, CanBuildResult
from asme.metrics.container.metrics_sampler import FixedItemsSampler
from asme.metrics.container.metrics_container import RankingMetricsContainer
from asme.utils.ioutils import load_file_with_item_ids
from data import CURRENT_SPLIT_PATH_CONTEXT_KEY


class FixedItemsMetricsFactory(ObjectFactory):

    """
    a factory to build the fixed items metrics container
    """

    KEY = 'fixed'

    def __init__(self):
        super().__init__()
        self.metrics_factory = MetricsFactory()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return require_config_keys(config, ['metrics', 'item_file'])

    def build(self, config: Config, context: Context) -> RankingMetricsContainer:
        metrics = self.metrics_factory.build(config.get_config(self.metrics_factory.config_path()), context)

        # Try to infer the item_file base_path if it was not absolute
        split_path = context.get(CURRENT_SPLIT_PATH_CONTEXT_KEY)
        infer_base_path(config, "item_file", split_path)
        items = load_file_with_item_ids(Path(config.get("item_file")))

        sampler = FixedItemsSampler(items)
        return RankingMetricsContainer(metrics, sampler)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
