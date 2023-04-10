from asme.core.main import *
from asme.data.datamodule.config import PreprocessingConfigProvider
from asme.data.datamodule.registry import register_preprocessing_config_provider
from asme.data.datasets.config import get_coveo_preprocessing_config
from plp_paper.movielens_extension import get_movielens_extended_preprocessing_config

torch.multiprocessing.set_start_method('spawn', force=True)

register_preprocessing_config_provider("ml-1m-extended",
                                       PreprocessingConfigProvider(get_movielens_extended_preprocessing_config,
                                                                   prefix="ml-1m-extended"
                                                                   ))

register_preprocessing_config_provider("ml-20m-extended",
                                       PreprocessingConfigProvider(get_movielens_extended_preprocessing_config,
                                                                   prefix="ml-20m-extended"
                                                                   ))

register_preprocessing_config_provider("coveo-extended",
                                       PreprocessingConfigProvider(get_coveo_preprocessing_config,
                                                                   prefix="coveo-extended"
                                                                   ))


if __name__ == "__main__":
    app()