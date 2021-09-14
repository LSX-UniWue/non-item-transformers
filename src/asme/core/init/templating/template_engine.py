from typing import Dict, Any, List

from asme.core.init.templating import TEMPLATES_CONFIG_KEY
from asme.core.init.templating.datasources.mask import MaskDataSourcesTemplateProcessor
from asme.core.init.templating.datasources.next import NextSequenceStepDataSourcesTemplateProcessor
from asme.core.init.templating.datasources.plain_training import PlainTrainingSourcesTemplateProcessor
from asme.core.init.templating.datasources.par_positive_negative import ParameterizedPositiveNegativeDataSourcesTemplateProcessor
from asme.core.init.templating.datasources.positive_negative import PositiveNegativeDataSourcesTemplateProcessor
from asme.core.init.templating.datasources.sliding_window import SlidingWindowDataSourceTemplateProcessor
from asme.core.init.templating.template_processor import TemplateProcessor
from asme.core.init.templating.trainer.output_directory import OutputDirectoryProcessor


class TemplateEngine:
    """
    An engine the consecutively applies template processors to the configuration.
    """

    def __init__(self,
                 head_processors: List[TemplateProcessor] = None,
                 tail_processors: List[TemplateProcessor] = None
                 ):
        super().__init__()
        if head_processors is None:
            head_processors = []

        if tail_processors is None:
            tail_processors = []
        self._template_processors = head_processors + [MaskDataSourcesTemplateProcessor(),
                                                       PositiveNegativeDataSourcesTemplateProcessor(),
                                                       ParameterizedPositiveNegativeDataSourcesTemplateProcessor(),
                                                       NextSequenceStepDataSourcesTemplateProcessor(),
                                                       PlainTrainingSourcesTemplateProcessor(),
                                                       SlidingWindowDataSourceTemplateProcessor(),
                                                       OutputDirectoryProcessor()] + tail_processors

    def modify(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply all template processors (in order) to the configuration.

        :param config: a config containing templates.
        :return: the updated config after applying all processors.
        """
        for template_processors in self._template_processors:
            if template_processors.can_modify(config):
                config = template_processors.modify(config)

        if TEMPLATES_CONFIG_KEY in config:
            # after all templates are applied remove the templates element from the config
            # check if every
            config.pop(TEMPLATES_CONFIG_KEY)
        return config

    def add_processor(self, processor: TemplateProcessor):
        """
        Adds a processor to the end of the list of applied processors.

        :param processor: a processor.
        :return:
        """
        self._template_processors.append(processor)
