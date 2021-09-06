from typing import Dict, Any

from asme.init.templating import TEMPLATES_CONFIG_KEY
from asme.init.templating.config_utils import config_entry_exists, get_config_value, set_config_value
from asme.init.templating.template_processor import TemplateProcessor


CHECKPOINT_PATH = ['trainer', 'checkpoint']
LOGGER_PATH = ["trainer", "loggers"]
LOGGER_TYPE_PATH = LOGGER_PATH + ['type']


class OutputDirectoryProcessor(TemplateProcessor):
    """
    Sets all output directories and files based on a common output directory.

    The processor is enabled if the following section exists in the configuration:
    ```
    {
       ...
        output_directory: "a custom path"
       ...
    }
    ```
    Modifies:
    1) the trainer.checkpoint section and sets `dirpath` to `{output_directory}/checkpoints` iff trainer.checkpoint is
       defined and no dirpath is set.

    2) the trainer.logger section and sets `save_dir` to {output_directory} if the logger is of type mlflow or
       tensorboard or csv.
       If no logger is specified a tensorboard logger will be created with `save_dir` set to {output_directory}.
    """
    def can_modify(self, config: Dict[str, Any]) -> bool:
        """
        Checks if the processor can be run. The following conditions must be met:
        1) `output_directory` must be present in the config file
        2) the user did not specify either `trainer.logger.save_dir` or `trainer.checkpoint.dirpath`.

        :param config: the configuration.

        :return: True if the processor can modify the configuration, False otherwise.
        """
        if TEMPLATES_CONFIG_KEY not in config:
            return False

        template_config = config.get(TEMPLATES_CONFIG_KEY)

        template_present = "unified_output" in template_config

        # check if user specified output directories on her own
        if config_entry_exists(config, LOGGER_PATH + ["save_dir"]) or\
                config_entry_exists(config, CHECKPOINT_PATH + ["dirpath"]):
            return False

        return template_present

    def modify(self, config: Dict[str, Any]) -> Dict[str, Any]:
        template_config = config.get('templates')
        unified_output_config = template_config.get('unified_output')
        output_base_dir = unified_output_config.get('path')

        self._modify_loggers(config, output_base_dir)
        self._modify_checkpoint(config, output_base_dir)

        return config

    def _modify_loggers(self,
                        config: Dict[str, Any],
                        output_base_dir: str):
        logger_element_exists = config_entry_exists(config, LOGGER_PATH)

        if logger_element_exists:
            loggers_config = get_config_value(config, LOGGER_PATH)
            if len(loggers_config) > 0:
                # TODO refactor logic logger-type wise
                for logger_type in loggers_config.keys():
                    if logger_type == "tensorboard" or logger_type == "mlflow" or logger_type == "csv":
                        set_config_value(config, LOGGER_PATH + [logger_type, "save_dir"], output_base_dir)
                    else:
                        print(f"Unknown logger {logger_type} I did not set the output directory!!")

                    # for tensorboard set version & name to empty string ("") to avoid subdirectories
                    if logger_type == "tensorboard" or logger_type == "csv":
                        logger_name_path = LOGGER_PATH + [logger_type, "name"]
                        if not config_entry_exists(config, logger_name_path):
                            set_config_value(config, logger_name_path, "")

                        version_name_path = LOGGER_PATH + [logger_type, "version"]
                        if not config_entry_exists(config, version_name_path):
                            set_config_value(config, version_name_path, "")
            else:
                # configure a default tensorboard logger
                set_config_value(config,
                                 LOGGER_PATH + ["tensorboard"],
                                 {"save_dir": output_base_dir, "name": "", "version": ""})

    def _modify_checkpoint(self, config: Dict[str, Any], output_base_dir: str):
        output_checkpoint_dir = f"{output_base_dir}/checkpoints"

        checkpoint_element_exists = config_entry_exists(config, CHECKPOINT_PATH)

        if checkpoint_element_exists:
            set_config_value(config, CHECKPOINT_PATH + ["dirpath"], output_checkpoint_dir)
