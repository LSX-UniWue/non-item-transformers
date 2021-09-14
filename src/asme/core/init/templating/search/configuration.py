from pathlib import Path
from typing import Dict, Any

from optuna import Trial

from asme.core.init.config_keys import TRAINER_CONFIG_KEY, CHECKPOINT_CONFIG_KEY, LOGGERS_CONFIG_KEY, \
    LOGGERS_TENSORBOARD_CONFIG_KEY, CHECKPOINT_CONFIG_DIR_PATH, CHECKPOINT_CONFIG_MONITOR, LOGGERS_CONFIG_VERSION, \
    LOGGERS_CONFIG_NAME, LOGGERS_CSV_CONFIG_KEY
from asme.core.init.templating.template_processor import TemplateProcessor
from asme.core.utils.run_utils import OBJECTIVE_METRIC_KEY


class SearchConfigurationTemplateProcessor(TemplateProcessor):
    """
    A template processor used for search, that configures the paths of the checkpoint callback and loggers using the
    provided trail:
    1. sets the path of the checkpoint to checkpoint.dirpath.parent / study_STUDY_NAME / TRAIL_NUMBER / checkpoint.dirpath.stem
    2. sets the path of all loggers to save_dir / study_STUDY_NAME / TRAIL_NUMBER
    """
    def __init__(self,
                 trial: Trial):
        super().__init__()
        self._trial = trial

    def can_modify(self, config: Dict[str, Any]) -> bool:
        return True

    def modify(self, config: Dict[str, Any]) -> Dict[str, Any]:
        study = self._trial.study
        study_name = f"study_{study.study_name}"
        trial_number = self._trial.number
        sub_path = Path(f"{study_name}") / f"{trial_number}"

        trainer_config = config.get(TRAINER_CONFIG_KEY)
        checkpoint_config = trainer_config.get(CHECKPOINT_CONFIG_KEY)
        checkpoint_path = Path(checkpoint_config[CHECKPOINT_CONFIG_DIR_PATH])
        checkpoint_path_parent = checkpoint_path.parent
        checkpoint_config[CHECKPOINT_CONFIG_DIR_PATH] = str(Path(checkpoint_path_parent) / sub_path / checkpoint_path.stem)
        objective_metric = study.user_attrs.get(OBJECTIVE_METRIC_KEY)
        if CHECKPOINT_CONFIG_MONITOR in checkpoint_config and checkpoint_config[CHECKPOINT_CONFIG_MONITOR] is not objective_metric:
            print(f'WARNING: monitor key not equal to objective metric, setting monitor to {objective_metric}')
        checkpoint_config[CHECKPOINT_CONFIG_MONITOR] = objective_metric

        loggers_config = trainer_config.get(LOGGERS_CONFIG_KEY, None)

        if loggers_config is not None:
            for logger_key, logger_config in loggers_config.items():
                if logger_key in [LOGGERS_TENSORBOARD_CONFIG_KEY, LOGGERS_CSV_CONFIG_KEY]:
                    logger_config[LOGGERS_CONFIG_VERSION] = str(trial_number)  # to remove the prefix version_
                    logger_config[LOGGERS_CONFIG_NAME] = study_name

        return config
