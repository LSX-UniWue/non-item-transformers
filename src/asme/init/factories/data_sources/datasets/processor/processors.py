from typing import List

from asme.init.factories.data_sources.datasets.processor.target_extractor import TargetExtractorProcessorFactory
from asme.init.factories.data_sources.datasets.processor.no_target_extractor import NoTargetExtractorProcessorFactory
from data.datasets.processors.processor import Processor
from asme.init.config import Config
from asme.init.context import Context
from asme.init.factories.common.conditional_based_factory import ConditionalFactory
from asme.init.factories.common.config_based_factory import ListFactory
from asme.init.factories.data_sources.datasets.processor.cloze_mask import ClozeProcessorFactory
from asme.init.factories.data_sources.datasets.processor.cut_to_fixed_sequence_length import \
    CutToFixedSequenceLengthProcessorFactory
from asme.init.factories.data_sources.datasets.processor.last_item_mask import LastItemMaskProcessorFactory
from asme.init.factories.data_sources.datasets.processor.par_pos_neg_sampler import \
    ParameterizedPositiveNegativeSamplerProcessorFactory
from asme.init.factories.data_sources.datasets.processor.pos_neg_sampler import PositiveNegativeSamplerProcessorFactory
from asme.init.factories.data_sources.datasets.processor.position_token import PositionTokenProcessorFactory
from asme.init.factories.data_sources.datasets.processor.tokenizer import TokenizerProcessorFactory
from asme.init.object_factory import ObjectFactory, CanBuildResult

FIXED_SEQUENCE_LENGTH_PROCESSOR_KEY = 'fixed_sequence_length_processor'


class ProcessorsFactory(ObjectFactory):
    KEY = 'processors'

    def __init__(self):
        super().__init__()

        # FIXME: register other processors
        self.processors_factories = ListFactory(
            ConditionalFactory('type', {
                'cloze': ClozeProcessorFactory(),
                'pos_neg': PositiveNegativeSamplerProcessorFactory(),
                'par_pos_neg': ParameterizedPositiveNegativeSamplerProcessorFactory(),
                'last_item_mask': LastItemMaskProcessorFactory(),
                'position_token': PositionTokenProcessorFactory(),
                'tokenizer': TokenizerProcessorFactory(),
                'target_extractor': TargetExtractorProcessorFactory(),
                'no_target_extractor': NoTargetExtractorProcessorFactory(),
                FIXED_SEQUENCE_LENGTH_PROCESSOR_KEY: CutToFixedSequenceLengthProcessorFactory()
            })
        )

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        return self.processors_factories.can_build(config, context)

    def build(self,
              config: Config,
              context: Context
              ) -> List[Processor]:
        return self.processors_factories.build(config, context)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return ['processors']

    def config_key(self) -> str:
        return self.KEY
