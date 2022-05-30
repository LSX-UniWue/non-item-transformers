from typing import List, Any

from asme.core.init.context import Context
from asme.core.init.factories import BuildContext

from asme.core.init.factories.common.dependencies_factory import DependenciesFactory
from asme.core.init.factories.common.list_elements_factory import NamedListElementsFactory
from asme.core.init.factories.features.tokenizer_factory import TokenizerFactory
from asme.core.init.factories.util import infer_whole_path, can_build_with_subsection, build_with_subsection
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.data import CURRENT_SPLIT_PATH_CONTEXT_KEY, DATASET_PREFIX_CONTEXT_KEY
from asme.data.datasets.sequence import MetaInformation


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

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        dependencies_result = can_build_with_subsection(self._dependencies, build_context)
        if dependencies_result.type != CanBuildResultType.CAN_BUILD:
            return dependencies_result

    def build(self, build_context: BuildContext) -> MetaInformation:
        config = build_context.get_current_config_section()
        context = build_context.get_context()
        feature_name = config.base_path[-1]
        feature_type = config.get_or_default('type', 'str')
        feature_is_sequence = config.get_or_default('sequence', True)
        column_name = config.get('column_name')
        sequence_length = config.get('sequence_length')
        run_tokenization = config.get_or_default('run_tokenization', True)

        feature_config = {}

        if run_tokenization:
            # If no explicit location for the vocabulary was provided, try to infer it
            split_path = context.get(CURRENT_SPLIT_PATH_CONTEXT_KEY)
            prefix = context.get(DATASET_PREFIX_CONTEXT_KEY)
            vocabulary_file = f"{prefix}.vocabulary.{column_name}.txt"
            infer_whole_path(config, ["tokenizer", "vocabulary", "file"], split_path, vocabulary_file)
            tokenizer = build_with_subsection(self._dependencies, build_context)['tokenizer']
        else:
            tokenizer = None

        for key in config.get_keys():
            if key not in self.CONFIG_KEYS:
                feature_config[key] = config.get(key)
        return MetaInformation(feature_name, feature_type, tokenizer=tokenizer, is_sequence=feature_is_sequence,
                               column_name=column_name, configs=feature_config, sequence_length=sequence_length)

    def is_required(self, build_context: BuildContext) -> bool:
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

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        return can_build_with_subsection(self.meta_information_factory, build_context)

    def build(self, build_context: BuildContext) -> Any:
        return build_with_subsection(self.meta_information_factory, build_context)

    def is_required(self, build_context: BuildContext) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
