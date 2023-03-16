from pathlib import Path
from typing import List

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.factories.metrics.metrics import MetricsFactory
from asme.core.init.factories.util import require_config_keys, infer_base_path, build_with_subsection
from asme.core.init.object_factory import ObjectFactory, CanBuildResult
from asme.core.metrics.container.metrics_sampler import FixedItemsSampler
from asme.core.metrics.container.metrics_container import RankingMetricsContainer
from asme.core.utils.ioutils import load_filtered_vocabulary
from asme.data import CURRENT_SPLIT_PATH_CONTEXT_KEY


class FixedItemsMetricsFactory(ObjectFactory):

    """
    a factory to build the fixed items metrics container
    """

    KEY = 'fixed'

    def __init__(self):
        super().__init__()
        self.metrics_factory = MetricsFactory()

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        return require_config_keys(build_context.get_current_config_section(), ['metrics', 'item_file'])

    def build(self, build_context: BuildContext) -> RankingMetricsContainer:
        metrics = build_with_subsection(self.metrics_factory, build_context)

        # Try to infer the item_file base_path if it was not absolute
        split_path = build_context.get_context().get(CURRENT_SPLIT_PATH_CONTEXT_KEY)
        current_config_section = build_context.get_current_config_section()
        #infer_base_path(current_config_section, "item_file", split_path)
        original_vocabulary = build_context.get_config().get_config(["features","item","tokenizer","vocabulary"]).get_or_default("file", None)
        if original_vocabulary is None:
            original_vocabulary = current_config_section.get_or_raise("model_vocabulary", "Please add the models vocabulary file as 'model_vocabulary' for fixed metric caluctlations manually for studies.")
        filter_file = current_config_section.get_or_raise("item_file", "Please set 'item_file' to the reduced vocabulary file.")
        items = load_filtered_vocabulary(Path(filter_file), Path(original_vocabulary))

        sampler = FixedItemsSampler(items)
        return RankingMetricsContainer(metrics, sampler)

    def is_required(self, build_context: BuildContext) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
