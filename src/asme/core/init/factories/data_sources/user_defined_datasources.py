import copy
from typing import List, Any, Dict

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories.data_sources.common import build_default_loader_config, set_path_based_on_split
from asme.core.init.factories.data_sources.loader import LoaderFactory
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.core.init.templating.datasources.datasources import Stage, DatasetSplit, NextPositionDatasetBuilder, \
    SequenceDatasetRatioSplitBuilder, NextPositionWindowDatasetBuilder, LeaveOneOutSessionDatasetBuilder, \
    LeaveOneOutSequenceWindowDatasetBuilder, LeaveOneOutNextPositionDatasetBuilder
from asme.data import CURRENT_SPLIT_PATH_CONTEXT_KEY


class UserDefinedDataSourcesFactory(ObjectFactory):
    DATASET_BUILDER = {
        DatasetSplit.RATIO_SPLIT: {
            "next_item": NextPositionDatasetBuilder(),
            "session": SequenceDatasetRatioSplitBuilder(),
            "window": NextPositionWindowDatasetBuilder()
        },
        DatasetSplit.LEAVE_ONE_OUT: {
            "next_item": LeaveOneOutNextPositionDatasetBuilder(),
            "session": LeaveOneOutSessionDatasetBuilder(),
            "window": LeaveOneOutSequenceWindowDatasetBuilder()
        }
    }

    def __init__(self):
        super().__init__()
        self._factory = LoaderFactory()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        for stage in Stage:
            result = self._factory.can_build(config.get_config(stage.value), context)
            if result.type != CanBuildResultType.CAN_BUILD:
                return result

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> Dict[str, Any]:

        # If no dataset path was specified, try to the use the one provided by the datamodule
        split = DatasetSplit[config.get("split").upper()]
        set_path_based_on_split(config, context, split)

        objects = {}
        for stage in Stage:
            # Use default values from the configuration and override them with values provided for the specific stage
            stage_config = _build_stage_config(config, stage)
            dataset_builder = self.DATASET_BUILDER[split][stage_config.pop("type")]
            # When using user defined data sources, all processors except the tokenizer have to specified explicitly
            processor_config = stage_config.pop("processors")

            # Generate the loader config similar to the template data sources
            loader_config = build_default_loader_config(stage_config, stage, [dataset_builder], processor_config)
            objects[stage.value] = self._factory.build(loader_config, context)

        return objects

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return ["datasources"]

    def config_key(self) -> str:
        return "datasources"


def _build_stage_config(config: Config, stage: Stage) -> Config:
    """
    Generates the config for a specific stage by merging the values of the parent config with those of the stage
    specific one. If a key is present in both configs, the value of the stage specific config takes precedence.

    :param config: The config including stage specific data as well as general information that might be used to
                   substitute values missing from the stage specific one.
    :param stage: The stage the config should be built for.

    :return: The stage specific configuration.
    """
    overwritten_stage_values = config.get_config([stage.value])
    default_values = _exclude_keys(config, [s.value for s in Stage])
    for key in overwritten_stage_values.get_keys():
        default_values.set(key, overwritten_stage_values.get(key))

    return default_values


def _exclude_keys(config: Config, keys_to_exclude: List[str]) -> Config:
    """
    Deepcopies the config but excludes the specified keys.

    :param config: The input config to copy.
    :param keys_to_exclude: The keys to exclude in the new config.

    :return: A config containing all original keys but the specified ones.
    """
    new_config = copy.deepcopy(config)
    for key in keys_to_exclude:
        new_config.pop(key)
    return new_config