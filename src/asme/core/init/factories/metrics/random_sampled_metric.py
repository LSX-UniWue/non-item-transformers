from typing import List

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories.metrics.metrics import MetricsFactory
from asme.core.init.factories.features.tokenizer_factory import get_tokenizer_key_for_voc, ITEM_TOKENIZER_ID
from asme.core.init.factories.util import require_config_keys
from asme.core.init.object_factory import ObjectFactory, CanBuildResult
from asme.core.metrics.container.metrics_sampler import NegativeMetricsSampler
from asme.core.metrics.container.metrics_container import RankingMetricsContainer
from asme.core.tokenization.tokenizer import Tokenizer


def _build_random_samples(tokenizer: Tokenizer) -> List[float]:
    special_token_ids = tokenizer.get_special_token_ids()
    return [0.0 if token_id in special_token_ids else 1.0 for token, token_id in tokenizer.vocabulary]


class RandomSampledMetricsFactory(ObjectFactory):
    """
    a factory to build the sampling metrics container that samples random over the item space
    """

    KEY = 'random_negative_sampled'

    def __init__(self):
        super().__init__()
        self.metrics_factory = MetricsFactory()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return require_config_keys(config, ['metrics', 'num_negative_samples'])

    def build(self, config: Config, context: Context) -> RankingMetricsContainer:
        metrics = self.metrics_factory.build(config.get_config(self.metrics_factory.config_path()), context)
        sample_size = config.get('num_negative_samples')
        item_tokenizer = context.get(get_tokenizer_key_for_voc(ITEM_TOKENIZER_ID))
        weights = _build_random_samples(item_tokenizer)

        sampler = NegativeMetricsSampler(weights, sample_size, 'random_negative_sampled')
        return RankingMetricsContainer(metrics, sampler)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
