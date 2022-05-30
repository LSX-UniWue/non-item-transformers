from typing import Union, Any, Dict, List, Type

from aim.sdk.adapters.pytorch_lightning import AimLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger, WandbLogger, LightningLoggerBase, CSVLogger

from asme.core.callbacks.best_model_writing_model_checkpoint import BestModelWritingModelCheckpoint
from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.factories.common.dependencies_factory import DependenciesFactory
from asme.core.init.factories.util import require_config_keys, build_with_subsection
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.core.init.trainer_builder import TrainerBuilder
from asme.core.utils.logging import get_logger


class KwargsFactory(ObjectFactory):

    def __init__(self,
                 class_type: Type,
                 key: str
                 ):
        super().__init__()
        self.class_type = class_type
        self.key = key

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, build_context: BuildContext) -> Union[Any, Dict[str, Any], List[Any]]:
        return self.class_type(**build_context.get_current_config_section().config)

    def is_required(self, build_context: BuildContext) -> bool:
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

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        config = build_context.get_current_config_section()
        result = require_config_keys(config, ["save_dir"])
        if result.type != CanBuildResultType.CAN_BUILD:
            return result

        if config.get("type") != "tensorboard":
            return CanBuildResult(CanBuildResultType.INVALID_CONFIGURATION, f"Can't build for type {config.get('type')}")

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, build_context: BuildContext) -> Union[Any, Dict[str, Any], List[Any]]:
        config = build_context.get_current_config_section()
        config_keys = config.get_keys()
        kwargs = {key: config.get(key) for key in config_keys if key != "type"}

        if "version" not in kwargs:
            kwargs["version"] = ""

        return TensorBoardLogger(**kwargs)

    def is_required(self, build_context: BuildContext) -> bool:
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


class AimLoggerFactory(KwargsFactory):
    def __init__(self):
        super().__init__(class_type=AimLogger, key="aim")


class CSVLoggerFactory(KwargsFactory):

    def __init__(self):
        super().__init__(class_type=CSVLogger, key="csv")

logger = get_logger()


class CheckpointFactory(ObjectFactory):
    def __init__(self, weights_only: bool = False):
        super().__init__()
        self.weights_only = weights_only

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, build_context: BuildContext) -> Union[Any, Dict[str, Any], List[Any]]:
        config = build_context.get_current_config_section()
        if not config.has_path("filename"):
            monitored_metric = config.get("monitor")
            config.set("filename", "{epoch}-" + f"{{{monitored_metric}}}")
        if not config.has_path("save_last"):
            config.set("save_last", True)

        if self.weights_only:
            config.set("save_weights_only", True)

            dirpath = config.get("dirpath")
            if dirpath is None or dirpath == "":
                logger.error(f"You need to set `dirpath` for Checkpoint callbacks!")

        wrapped_checkpoint = BestModelWritingModelCheckpoint(**config.config)
        return wrapped_checkpoint

    def is_required(self, build_context: BuildContext) -> bool:
        return True

    def config_path(self) -> List[str]:
        return ["checkpoint-weights-only"] if self.weights_only else ["checkpoint"]

    def config_key(self) -> str:
        return "checkpoint-weights-only" if self.weights_only else "checkpoint"




class EarlyStoppingCallbackFactory(KwargsFactory):

    def __init__(self):
        super().__init__(class_type=EarlyStopping, key='early_stopping')


class LoggersFactory(ObjectFactory):

    def __init__(self):
        super().__init__()
        self.dependency_factors = DependenciesFactory([TensorboardLoggerFactory(),
                                                       MLFlowLoggerFactory(),
                                                       WandBLoggerFactory(),
                                                       CSVLoggerFactory(),
                                                       AimLoggerFactory()],
                                                      optional_based_on_path=True)

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        return self.dependency_factors.can_build(build_context)

    def build(self, build_context: BuildContext) -> List[LightningLoggerBase]:
        loggers_dict = self.dependency_factors.build(build_context)
        return list(loggers_dict.values())

    def is_required(self, build_context: BuildContext) -> bool:
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
            LoggersFactory(),
            CheckpointFactory(),
            CheckpointFactory(weights_only=True),
            EarlyStoppingCallbackFactory()
        ], optional_based_on_path=True)

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, build_context: BuildContext) -> TrainerBuilder:
        config = build_context.get_current_config_section()
        config_keys = config.get_keys()
        dependency_keys = self.dependencies.get_dependency_keys()

        dependencies = build_with_subsection(self.dependencies, build_context)

        trainer_params_names = [x for x in config_keys if x not in dependency_keys]
        trainer_params = {key: config.get(key) for key in trainer_params_names}

        trainer_builder = TrainerBuilder(trainer_parameters=trainer_params)
        trainer_builder.add_logger(dependencies["loggers"])

        trainer_builder.add_callback(dependencies["checkpoint"])

        if "checkpoint-weights-only" in dependencies:
            trainer_builder.add_callback(dependencies["checkpoint-weights-only"])

        # add optional early stopping
        early_stopping_callback = dependencies.get('early_stopping', None)
        if early_stopping_callback is not None:
            trainer_builder.add_callback(early_stopping_callback)

        return trainer_builder

    def is_required(self, build_context: BuildContext) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
