from typing import List

from asme.core.init.config import Config
from asme.core.init.container import Container
from asme.core.init.context import Context
from asme.core.init.factories.common.conditional_based_factory import ConditionalFactory
from asme.core.init.factories.common.dependencies_factory import DependenciesFactory
from asme.core.init.factories.data_sources.datamodule import DataModuleFactory
from asme.core.init.factories.features.features_factory import FeaturesFactory
from asme.core.init.factories.features.tokenizer_factory import TOKENIZERS_PREFIX
from asme.core.init.factories.include.import_factory import ImportFactory
from asme.core.init.factories.trainer import TrainerBuilderFactory
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
                TrainerBuilderFactory()
            ]
        )

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:

        import_config = config.get_config(self.import_factory.config_path())
        can_build_result = self.import_factory.can_build(import_config, context)

        if can_build_result.type != CanBuildResultType.CAN_BUILD:
            return can_build_result

        datamodule_config = config.get_config(self.datamodule_factory.config_path())
        can_build_result = self.datamodule_factory.can_build(datamodule_config, context)

        if can_build_result.type != CanBuildResultType.CAN_BUILD:
            return can_build_result

        can_build_result = self.dependencies.can_build(config, context)

        if can_build_result.type != CanBuildResultType.CAN_BUILD:
            return can_build_result

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> Container:

        # First, we have to load all additional modules
        self.import_factory.build(config, context)

        # Then we build the datamodule such that we can invoke preprocessing
        datamodule_config = config.get_config(self.datamodule_factory.config_path())
        datamodule = self.datamodule_factory.build(datamodule_config, context)
        context.set(self.datamodule_factory.config_path(), datamodule)
        # Preprocess the dataset
        datamodule.prepare_data()

        features_config = config.get_config(self.features_factory.config_path())
        meta_information = list(self.features_factory.build(features_config, context).values())
        context.set(features_config.base_path, meta_information)
        for info in meta_information:
            if info.tokenizer is not None:
                context.set([TOKENIZERS_PREFIX, info.feature_name], info.tokenizer)

        all_dependencies = self.dependencies.build(config, context)

        for key, object in all_dependencies.items():
            if isinstance(object, dict):
                for section, o in object.items():
                    context.set([key, section], o)
            else:
                context.set(key, object)

        return Container(context.as_dict())

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return ""
