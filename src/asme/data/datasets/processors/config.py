from asme.core.init.factories.data_sources.datasets.processor.cloze_mask import ClozeProcessorFactory
from asme.core.init.factories.data_sources.datasets.processor.cut_to_fixed_sequence_length import \
    CutToFixedSequenceLengthProcessorFactory
from asme.core.init.factories.data_sources.datasets.processor.last_item_mask import LastItemMaskProcessorFactory
from asme.core.init.factories.data_sources.datasets.processor.negative_item_sampler import \
    NegativeItemSamplerProcessorFactory
from asme.core.init.factories.data_sources.datasets.processor.no_target_extractor import \
    NoTargetExtractorProcessorFactory
from asme.core.init.factories.data_sources.datasets.processor.pad_feature import PadFeatureProcessorFactory
from asme.core.init.factories.data_sources.datasets.processor.pos_neg_sampler import \
    PositiveNegativeSamplerProcessorFactory
from asme.core.init.factories.data_sources.datasets.processor.position_token import PositionTokenProcessorFactory
from asme.core.init.factories.data_sources.datasets.processor.positive_item_extractor import \
    PositiveItemExtractorProcessorFactory
from asme.core.init.factories.data_sources.datasets.processor.input_dictionary import InputDictionaryProcessorFactory
from asme.core.init.factories.data_sources.datasets.processor.target_extractor import TargetExtractorProcessorFactory
from asme.core.init.factories.data_sources.datasets.processor.tokenizer import TokenizerProcessorFactory
from asme.data.datasets.processors.registry import register_processor, ProcessorConfig

register_processor("cloze", ProcessorConfig(ClozeProcessorFactory))
register_processor("pos_neg", ProcessorConfig(PositiveNegativeSamplerProcessorFactory))
register_processor("last_item_mask", ProcessorConfig(LastItemMaskProcessorFactory))
register_processor("position_token", ProcessorConfig(PositionTokenProcessorFactory))
register_processor("tokenizer", ProcessorConfig(TokenizerProcessorFactory))
register_processor("target_extractor", ProcessorConfig(TargetExtractorProcessorFactory))
register_processor("no_target_extractor", ProcessorConfig(NoTargetExtractorProcessorFactory))
register_processor("negative_item_sampler", ProcessorConfig(NegativeItemSamplerProcessorFactory))
register_processor("positive_item_extractor", ProcessorConfig(PositiveItemExtractorProcessorFactory))
register_processor("fixed_sequence_length_processor", ProcessorConfig(CutToFixedSequenceLengthProcessorFactory))
register_processor("pad_feature", ProcessorConfig(PadFeatureProcessorFactory))
register_processor("input_dictionaries", ProcessorConfig(InputDictionaryProcessorFactory))

