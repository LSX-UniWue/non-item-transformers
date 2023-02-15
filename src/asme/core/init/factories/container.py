from typing import List, Any

from asme.core.init.container import Container
from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.factories.common.conditional_based_factory import ConditionalFactory
from asme.core.init.factories.common.dependencies_factory import DependenciesFactory
from asme.core.init.factories.data_sources.datamodule import DataModuleFactory
from asme.core.init.factories.features.features_factory import FeaturesFactory
from asme.core.init.factories.features.special_values_feature_factory import SPECIAL_VALUES_PREFIX
from asme.core.init.factories.features.tokenizer_factory import TOKENIZERS_PREFIX
from asme.core.init.factories.features.item_dictionary_factory import DICTIONARIES_PREFIX
from asme.core.init.factories.include.import_factory import ImportFactory
from asme.core.init.factories.trainer import TrainerBuilderFactory
from asme.core.init.factories.evaluation.evaluation import EvaluationFactory
from asme.core.init.factories.util import can_build_with_subsection, build_with_subsection
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.core.modules.registry import REGISTERED_MODULES


class ContainerFactory(ObjectFactory):
    def __init__(self):
        super().__init__()
        self.import_factory = ImportFactory()
        self.features_factory = FeaturesFactory()
        self.datamodule_factory = DataModuleFactory()
        self.dependencies = DependenciesFactory(
            [
                ConditionalFactory('type',
                                   REGISTERED_MODULES,
                                   config_key='module',
                                   config_path=['module']),
                TrainerBuilderFactory(),

            ]
        )
        self.evaluation_factory = DependenciesFactory([EvaluationFactory()], optional_based_on_path=True)

    def can_build(self,
                  build_context: BuildContext
                  ) -> CanBuildResult:

        can_build_result = can_build_with_subsection(self.import_factory, build_context)
        if can_build_result.type != CanBuildResultType.CAN_BUILD:
            return can_build_result

        can_build_result = can_build_with_subsection(self.features_factory, build_context)
        if can_build_result.type != CanBuildResultType.CAN_BUILD:
            return can_build_result

        can_build_result = can_build_with_subsection(self.datamodule_factory)
        if can_build_result.type != CanBuildResultType.CAN_BUILD:
            return can_build_result

        can_build_result = can_build_with_subsection(self.dependencies, build_context)
        if can_build_result.type != CanBuildResultType.CAN_BUILD:
            return can_build_result

        can_build_result = can_build_with_subsection(self.evaluation_factory, build_context)
        if can_build_result.type != CanBuildResultType.CAN_BUILD:
            return can_build_result

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              build_context: BuildContext
              ) -> Container:

        # First, we have to load all additional modules
        build_with_subsection(self.import_factory, build_context)

        # Then we build the datamodule such that we can invoke preprocessing
        datamodule = build_with_subsection(self.datamodule_factory, build_context)
        # Preprocess the dataset
        datamodule.prepare_data()
        build_context.get_context().set(self.datamodule_factory.config_key(), datamodule)

        def build_metainformation(factory: ObjectFactory, build_context: BuildContext) -> Any:
            meta_information = list(factory.build(build_context).values())
            build_context.get_context().set(build_context.get_current_config_section().base_path, meta_information)
            for info in meta_information:
                if info.tokenizer is not None:
                    build_context.get_context().set([TOKENIZERS_PREFIX, info.feature_name], info.tokenizer)
                if info.special_values is not None:
                    build_context.get_context().set([SPECIAL_VALUES_PREFIX, info.feature_name], info.special_values)

        build_with_subsection(self.features_factory, build_context, build_metainformation)

        all_dependencies = build_with_subsection(self.dependencies, build_context)

        for key, object in all_dependencies.items():
            if isinstance(object, dict):
                for section, o in object.items():
                    build_context.get_context().set([key, section], o)
            else:
                build_context.get_context().set(key, object)

        build_with_subsection(self.evaluation_factory, build_context)

        return Container(build_context.get_context().as_dict())


    def is_required(self, build_context: BuildContext) -> bool:
        return True

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return ""
