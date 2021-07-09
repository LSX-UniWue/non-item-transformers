from typing import List, Any

from asme.init.config import Config
from asme.init.context import Context

from asme.init.factories.common.dependencies_factory import DependenciesFactory
from asme.init.factories.common.list_elements_factory import NamedListElementsFactory
from asme.init.factories.features.tokenizer_factory import TokenizerFactory
from asme.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from data.datasets.sequence import MetaInformation


class MetaInformationFactory(ObjectFactory):
    """
    Builds a single meta information containing the required information of the feature for later
    parsing, padding,
    """
    KEY = "meta_information"

    CONFIG_KEYS = ['type', 'sequence', 'column_name', "tokenizer"]

    def __init__(self,
                 dependencies=DependenciesFactory([TokenizerFactory()])
                 ):
        super().__init__()
        self._dependencies = dependencies

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        dependencies_result = self._dependencies.can_build(config, context)
        if dependencies_result.type != CanBuildResultType.CAN_BUILD:
            return dependencies_result

    def build(self,
              config: Config,
              context: Context
              ) -> MetaInformation:
        feature_name = config.base_path[-1]
        feature_type = config.get_or_default('type', 'str')
        feature_is_sequence = config.get_or_default('sequence', True)
        column_name = config.get('column_name')
        sequence_length = config.get('sequence_length')

        feature_config = {}

        tokenizer = self._dependencies.build(config, context)['tokenizer']

        for key in config.get_keys():
            if key not in self.CONFIG_KEYS:
                feature_config[key] = config.get(key)
        return MetaInformation(feature_name, feature_type, tokenizer=tokenizer, is_sequence=feature_is_sequence,
                               column_name=column_name, configs=feature_config, sequence_length=sequence_length)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return self.KEY


class FeaturesFactory(ObjectFactory):
    """
    Builds all meta information within the `features` section.
    """

    KEY = "features"

    def __init__(self,
                 meta_information_factory: NamedListElementsFactory = NamedListElementsFactory(MetaInformationFactory())
                 ):
        super().__init__()
        self.meta_information_factory = meta_information_factory

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return self.meta_information_factory.can_build(config, context)

    def build(self, config: Config, context: Context) -> Any:
        return self.meta_information_factory.build(config, context)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
