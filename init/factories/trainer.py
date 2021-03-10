from typing import Union, Any, Dict, List, Type

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger, WandbLogger, LightningLoggerBase, CSVLogger

from init.config import Config
from init.context import Context
from init.factories.common.dependencies_factory import DependenciesFactory
from init.factories.util import require_config_keys
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from init.trainer_builder import TrainerBuilder


class KwargsFactory(ObjectFactory):

    def __init__(self,
                 class_type: Type,
                 key: str
                 ):
        super().__init__()
        self.class_type = class_type
        self.key = key

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        return self.class_type(**config.config)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.key]

    def config_key(self) -> str:
        return self.key


class TensorboardLoggerFactory(ObjectFactory):

    KEY = "tensorboard"
    REQUIRED_KEYS = ["save_dir"]

    def __init__(self):
        super(TensorboardLoggerFactory, self).__init__()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        result = require_config_keys(config, ["save_dir"])
        if result.type != CanBuildResultType.CAN_BUILD:
            return result

        if config.get("type") != "tensorboard":
            return CanBuildResult(CanBuildResultType.INVALID_CONFIGURATION, f"Can't build for type {config.get('type')}")

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        config_keys = config.get_keys()
        kwargs = {key: config.get(key) for key in config_keys if key != "type"}

        if "version" not in kwargs:
            kwargs["version"] = ""

        return TensorBoardLogger(**kwargs)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY


class MLFlowLoggerFactory(KwargsFactory):
    def __init__(self):
        super().__init__(class_type=MLFlowLogger, key="mlflow")


class WandBLoggerFactory(KwargsFactory):
    def __init__(self):
        super().__init__(class_type=WandbLogger, key="wandb")


class CSVLoggerFactory(KwargsFactory):
    
    def __init__(self):
        super().__init__(class_type=CSVLogger, key="csv")


class CheckpointFactory(KwargsFactory):

    def __init__(self):
        super().__init__(class_type=ModelCheckpoint, key="checkpoint")


class EarlyStoppingCallbackFactory(KwargsFactory):

    def __init__(self):
        super().__init__(class_type=EarlyStopping, key='early_stopping')


class LoggersFactors(ObjectFactory):

    def __init__(self):
        super().__init__()
        self.dependency_factors = DependenciesFactory([TensorboardLoggerFactory(),
                                                       MLFlowLoggerFactory(),
                                                       WandBLoggerFactory(),
                                                       CSVLoggerFactory()],
                                                      optional_based_on_path=True)

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return self.dependency_factors.can_build(config, context)

    def build(self, config: Config, context: Context) -> List[LightningLoggerBase]:
        loggers_dict = self.dependency_factors.build(config, context)
        return list(loggers_dict.values())

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return ["loggers"]

    def config_key(self) -> str:
        return "loggers"


class TrainerBuilderFactory(ObjectFactory):

    KEY = "trainer"

    def __init__(self):
        super().__init__()

        self.dependencies = DependenciesFactory([
            LoggersFactors(),
            CheckpointFactory(),
            EarlyStoppingCallbackFactory()
        ], optional_based_on_path=True)

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> TrainerBuilder:
        config_keys = config.get_keys()
        dependency_keys = self.dependencies.get_dependency_keys()

        dependencies = self.dependencies.build(config, context)

        trainer_params_names = [x for x in config_keys if x not in dependency_keys]
        trainer_params = {key: config.get(key) for key in trainer_params_names}

        trainer_builder = TrainerBuilder(trainer_parameters=trainer_params)
        trainer_builder.add_logger(dependencies["loggers"])
        trainer_builder.add_callback(dependencies["checkpoint"])

        # add optional early stopping
        early_stopping_callback = dependencies.get('early_stopping', None)
        if early_stopping_callback is not None:
            trainer_builder.add_callback(early_stopping_callback)

        return trainer_builder

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
