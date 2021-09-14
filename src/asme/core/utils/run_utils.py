import json
from pathlib import Path
from typing import List, Optional, Dict

import _jsonnet
import optuna
from optuna import Study
from optuna.study import StudyDirection
from pytorch_lightning.utilities import cloud_io
from torch.utils.data import DataLoader

from asme.core.init.config import Config
from asme.core.init.config_keys import TRAINER_CONFIG_KEY, CHECKPOINT_CONFIG_KEY, CHECKPOINT_CONFIG_DIR_PATH
from asme.core.init.container import Container
from asme.core.init.context import Context
from asme.core.init.factories.container import ContainerFactory
from asme.core.init.templating.template_engine import TemplateEngine
from asme.core.init.templating.template_processor import TemplateProcessor
from asme.core.tokenization.tokenizer import Tokenizer
from asme.core.utils import logging
from asme.core.utils.ioutils import PROCESSED_CONFIG_NAME, find_all_files
from asme.data.datasets import ITEM_SEQ_ENTRY_NAME
from asme.data.example_logging import ExampleLogger

logger = logging.get_logger(__name__)

""" key to retrieve the object metric used in hyperparameter study """
OBJECTIVE_METRIC_KEY = 'objective_metric'
""" key for retrieving the output path of the trail """
TRIAL_BASE_PATH = 'base_output_path'
""" the checkpoint file extension """
CHECKPOINT_FILE_EXTENSION = '.ckpt'


def load_config(config_file: Path,
                additional_head_processors: List[TemplateProcessor] = None,
                additional_tail_processors: List[TemplateProcessor] = None
                ) -> Config:
    if not config_file.exists():
        print(f"the config file cannot be found. Please check the path '{config_file}'!")
        exit(-1)
    if additional_head_processors is None:
        additional_head_processors = []
    if additional_tail_processors is None:
        additional_tail_processors = []

    config_json = _jsonnet.evaluate_file(str(config_file))

    loaded_config = json.loads(config_json)

    template_engine = TemplateEngine(head_processors=additional_head_processors,
                                     tail_processors=additional_tail_processors)

    config_to_use = template_engine.modify(loaded_config)
    return Config(config_to_use)


def load_hyperopt_config(hyperopt_config_file: Path, processors: List[TemplateProcessor]) -> Config:
    if not hyperopt_config_file.exists():
        print(f"the hyperopt config file is missing. Please check that {hyperopt_config_file} exists.")

    config_json = json.loads(_jsonnet.evaluate_file(str(hyperopt_config_file)))

    template_engine = TemplateEngine(tail_processors=processors)

    return Config(template_engine.modify(config_json))


def create_container(config: Config) -> Container:
    context = Context()

    container_factory = ContainerFactory()
    container = container_factory.build(config, context)

    return container


def load_container(config_file: Path) -> Container:
    config_raw = load_config(config_file)
    return create_container(config_raw)


def load_and_restore_container(config_file: Path,
                               checkpoint_file: Path,
                               gpus: int
                               ) -> Container:
    """
    loads the container and restores the state from the checkpoint file
    :param config_file:
    :param checkpoint_file:
    :param gpus:
    :return: the loaded and restored container
    """

    container = load_container(config_file)
    module = container.module()

    # FIXME: try to use load_from_checkpoint later
    # load checkpoint <- we don't use the PL function load_from_checkpoint because it does
    # not work with our module class system
    map_location = None if gpus > 0 else 'cpu'
    ckpt = cloud_io.load(checkpoint_file, map_location=map_location)

    # acquire state_dict
    state_dict = ckpt["state_dict"]

    # load parameters and freeze the model
    module.load_state_dict(state_dict)
    module.freeze()

    trainer_builder = container.trainer()
    trainer_builder.set("gpus", gpus)

    return container


def get_config_of_best_run_from_study(study: Study
                                      ) -> Path:
    best_trails = study.best_trials
    if len(best_trails) > 0:
        print('more than one best trail, using the first one')

    best_trail = best_trails[0]
    base_path = Path(best_trail.user_attrs.get(TRIAL_BASE_PATH))
    return base_path / PROCESSED_CONFIG_NAME


def _extract_metric_value_from_checkpoint_file(checkpoint_file: Path
                                               ) -> float:
    return float(checkpoint_file.name.split('=')[-1].replace(CHECKPOINT_FILE_EXTENSION, ''))


def get_checkpoint_file_from(config: Config,
                             study: Study
                             ) -> Path:
    """
    returns the path to the checkpoint path to the best trail of the study
    :param config: the config to use
    :param study: the study to use
    :return: the path to the checkpoint path
    """
    path = config.get([TRAINER_CONFIG_KEY, CHECKPOINT_CONFIG_KEY, CHECKPOINT_CONFIG_DIR_PATH])
    # we use the objective metric saved in the study, because we may change it while applying templates
    objective_metric = study.user_attrs.get(OBJECTIVE_METRIC_KEY)
    checkpoint_files = find_all_files(path, CHECKPOINT_FILE_EXTENSION)
    metrics_checkpoints_list = []
    for checkpoint_file in checkpoint_files:
        if objective_metric in str(checkpoint_file):
            metric_value = _extract_metric_value_from_checkpoint_file(checkpoint_file)
            metrics_checkpoints_list.append((metric_value, checkpoint_file))

    metrics_checkpoints_list_sorted = sorted(metrics_checkpoints_list, key=lambda metric_file: metric_file[0],
                                             reverse=study.direction == StudyDirection.MAXIMIZE)
    return metrics_checkpoints_list_sorted[0][1]


def load_and_restore_from_file_or_study(checkpoint_file: Optional[Path],
                                        config_file: Optional[Path],
                                        study_name: Optional[str],
                                        study_storage: Optional[str],
                                        gpus: Optional[int]
                                        ) -> Optional[Container]:
    # check if we can get a checkpoint file or not
    if (checkpoint_file is None or config_file is None) and (study_name is None or study_storage is None):
        return None

    if study_name is not None:
        study = optuna.study.load_study(study_name=study_name, storage=study_storage)
        config_file = get_config_of_best_run_from_study(study)
        config = load_config(config_file)
        checkpoint_file = get_checkpoint_file_from(config, study)
    return load_and_restore_container(config_file, checkpoint_file, gpus=gpus)


def log_dataloader_example(dataloader: DataLoader,
                           tokenizers: Dict[str, Tokenizer],
                           example_type: str
                           ) -> None:
    def _extract_tokenizer_key(example_key: str) -> str:
        return example_key.split('.')[0]

    dataset = dataloader.dataset
    if isinstance(dataset, ExampleLogger):
        logger.info(f'using {len(dataset)} examples for {example_type}')
        logger.info('----------------------------')
        logger.info(f"{example_type} example")
        logger.info('----------------------------')
        for example in dataset.get_data_examples(num_examples=1):
            logger.info('loaded sequence data')
            for key, value in example.sequence_data.items():
                logger.info(f'\t{key} = {value}')
            logger.info('processed data')
            for key, value in example.processed_data.items():
                logger.info(f'\t{key} = {value}')
                tokenizer_key = 'item' if key.startswith(ITEM_SEQ_ENTRY_NAME) else _extract_tokenizer_key(key)
                if tokenizer_key in tokenizers:
                    tokens = tokenizers[tokenizer_key].convert_ids_to_tokens(value)
                    logger.info(f'\t(tokens) {tokens}')
            logger.info('')
