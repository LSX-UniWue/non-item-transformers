import json
import os
from pathlib import Path
from typing import List, Callable, Any, Optional

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LoggerCollection, LightningLoggerBase

from asme.callbacks.best_model_writing_model_checkpoint import BestModelWritingModelCheckpoint
from asme.init.config import Config


PROCESSED_CONFIG_NAME = "config.jsonnet"
FINISHED_FLAG_NAME = ".FINISHED"


def save_config(config: Config,
                path: Path,
                make_parents=True
                ):
    """
    Saves the provided configuration at the specified path, optionally creating parent directories.
    """
    if make_parents:
        os.makedirs(path,exist_ok=True)
    full_path = path / PROCESSED_CONFIG_NAME
    with open(full_path, "w") as f:
        json.dump(config.config, f, indent=4)


def _build_log_dir_from_logger(logger: LightningLoggerBase) -> Optional[Path]:
    # FIXME: remove method as soon as the is a common build method for this in the lightning lib
    if logger.save_dir is None:
        return None

    log_dir = Path(logger.save_dir)
    logger_name = logger.name
    if logger_name and len(logger_name) > 0:
        log_dir = log_dir / logger_name

    logger_version = logger.version

    if logger_version is not None and len(logger_version) > 0:
        logger_version_subpath = logger_version if isinstance(logger_version, str) else f"version_{logger_version}"
        log_dir = log_dir / logger_version_subpath

    return log_dir


def determine_log_dir(trainer: Trainer) -> Path:
    log_dir = None

    # try to infer from the logger configuration
    logger = trainer.logger
    if type(logger) is LightningLoggerBase:
        log_dir = _build_log_dir_from_logger(logger)
    elif type(logger) is LoggerCollection:
        for l in logger._logger_iterable:
            dir = _build_log_dir_from_logger(l)
            if dir is not None:
                log_dir = dir
                break
    else:
        print(f"Unable to infer log_dir from logger configuration. Failing over to ModelCheckpoint.")

    # no logger provided a directory, e.g. mlflow is used
    # fail over to model_checkpoint and use the parent directory
    if log_dir is None:
        ckpt_callback = trainer.checkpoint_callback
        if ckpt_callback is not None:
            if type(ckpt_callback) is ModelCheckpoint:
                log_dir = Path(ckpt_callback.dirpath).parent
            elif type(ckpt_callback) is BestModelWritingModelCheckpoint:
                log_dir = Path(ckpt_callback.output_base_path).parent
            else:
                print(f"Could not infer output path from checkpoint callback of type: {type(ckpt_callback)}")

    # querying logger / checkpoint callback failed: fallback to Trainer.default_root_dir
    if log_dir is None:
        log_dir = Path(trainer.default_root_dir)

    print(log_dir)
    return log_dir


def save_finished_flag(path: Path,
                       make_parents=True
                       ):
    """
    Saves an empty file named '.FINISHED' at the specified path, optionally creating parent directories.
    """
    if make_parents:
        os.makedirs(path, exist_ok=True)
    full_path = path / FINISHED_FLAG_NAME
    with open(full_path, "w"):
        pass


def finished_flag_exists(path: Path) -> bool:
    """
    Checks whether a finished flag, i.e a file named '.FINISHED' exists at the provided path.
    """
    full_path = path / FINISHED_FLAG_NAME
    return os.path.isfile(full_path)


def find_all_files(base_path: Path,
                   suffix: str,
                   follow_links: bool = True
                   ) -> List[Path]:
    all_files = []
    for root, dirs, files in os.walk(str(base_path), followlinks=follow_links):
        for file in files:
            if file.endswith(suffix):
                all_files.append(Path(os.path.join(root, file)))

    return all_files


def load_file_with_item_ids(path: Path) -> List[int]:
    """
    loads a file containing item ids into a list
    :param path: the path of the file
    :return:
    """
    items = _load_file_line_my_line(path, int)
    sorted(items)
    return items


def _load_file_line_my_line(path: Path, line_converter: Callable[[str], Any]) -> List[Any]:
    with open(path) as item_file:
        return [line_converter(line) for line in item_file.readlines()]
