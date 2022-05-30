from asme.core.init.factories.data_sources.registry import register_template, TemplateConfig
from asme.core.init.factories.data_sources.template_datasources import MaskTemplateDataSourcesFactory, \
    PositiveNegativeTemplateDataSourcesFactory, NextSequenceStepTemplateDataSourcesFactory, \
    ParameterizedPositiveNegativeTemplateDataSourcesFactory, PlainTrainingTemplateDataSourcesFactory, \
    ParallelSeqTrainingTemplateDataSourcesFactory, SlidingWindowTemplateDataSourcesFactory

register_template("masked", TemplateConfig(MaskTemplateDataSourcesFactory))
register_template("pos_neg", TemplateConfig(PositiveNegativeTemplateDataSourcesFactory))
register_template("next_sequence_step", TemplateConfig(NextSequenceStepTemplateDataSourcesFactory))
register_template("par_pos_neg", TemplateConfig(ParameterizedPositiveNegativeTemplateDataSourcesFactory))
register_template("plain", TemplateConfig(PlainTrainingTemplateDataSourcesFactory))
register_template("par_seq", TemplateConfig(ParallelSeqTrainingTemplateDataSourcesFactory))
register_template("sliding_window", TemplateConfig(SlidingWindowTemplateDataSourcesFactory))
