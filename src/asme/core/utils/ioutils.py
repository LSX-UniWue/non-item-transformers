import json
import os

import pandas as pd
import yaml

from pathlib import Path
from loguru import logger
from typing import List, Callable, Any, Optional
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LoggerCollection, LightningLoggerBase
from aim.sdk.adapters.pytorch_lightning import AimLogger

from asme.core.callbacks.best_model_writing_model_checkpoint import BestModelWritingModelCheckpoint
from asme.core.init.config import Config

PROCESSED_CONFIG_NAME = "config"
FINISHED_FLAG_NAME = ".FINISHED"


def save_config(config: Config,
                source_file: Path,
                path: Path,
                make_parents=True
                ):
    """
    Saves the provided configuration at the specified path, optionally creating parent directories.
    """
    if make_parents:
        os.makedirs(path, exist_ok=True)

    suffix = source_file.suffix

    if suffix == ".json":
        full_path = path / f"{PROCESSED_CONFIG_NAME}{suffix}"
        with open(full_path, "w") as f:
            json.dump(config.config, f, indent=4)
    elif suffix == ".yaml" or suffix == ".yml":
        full_path = path / f"{PROCESSED_CONFIG_NAME}{suffix}"
        with open(full_path, "w") as f:
            yaml.dump(config.config, f)
    else:
        logger.warning(f"Config file {source_file.name} is not in json or yaml format, saving resolved config as yaml.")
        full_path = path / f"{PROCESSED_CONFIG_NAME}.yaml"
        with open(full_path, "w") as f:
            yaml.dump(config.config, f)


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
    if type(logger) is LightningLoggerBase and type(logger) is not AimLogger:
        log_dir = _build_log_dir_from_logger(logger)
    elif type(logger) is LoggerCollection:
        for l in logger._logger_iterable:

            # aim logger does not provide a log_dir
            # TODO provide an adapter for AimLogger that returns `None`
            if type(l) is AimLogger:
                continue

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
                log_dir = Path(ckpt_callback.dirpath).parent
            else:
                print(f"Could not infer output path from checkpoint callback of type: {type(ckpt_callback)}")

    # querying logger / checkpoint callback failed: fallback to Trainer.default_root_dir
    if log_dir is None:
        log_dir = Path(trainer.default_root_dir)

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


def load_filtered_vocabulary(vocab_file_reduced: Path, vocab_file_original: Path) -> List[int]:
    """
    Loads a restricted and the original vocabulary file.
    :param vocab_file_reduced: vocabulary file containing a reduced set
    :param vocab_file_original: original vocabulary file
    :return: a list of the item ids in the original vocabulary file which are also present in the reduced vocabulary
    """
    reduced_vocabulary = pd.read_csv(vocab_file_reduced, header= None, sep="\t")
    full_vocabulary = pd.read_csv(vocab_file_original, header= None, sep="\t")

    selected_vocabulary = full_vocabulary.merge(reduced_vocabulary, on=0, how='inner')
    return selected_vocabulary["1_x"].astype(int).tolist()

    #items = _load_file_line_my_line(path, int)
    #sorted(items)
    #return items


def _load_file_line_my_line(path: Path, line_converter: Callable[[str], Any]) -> List[Any]:
    with open(path) as item_file:
        return [line_converter(line) for line in item_file.readlines()]
