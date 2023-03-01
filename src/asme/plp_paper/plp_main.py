from asme.core.main import *

register_preprocessing_config_provider("ml-1m-extended",
                                       PreprocessingConfigProvider(get_ml_1m_extended_preprocessing_config,
                                                                   prefix="ml-1m-extended"
                                                                   ))

register_preprocessing_config_provider("ml-20m-extended",
                                       PreprocessingConfigProvider(get_ml_1m_extended_preprocessing_config,
                                                                   prefix="ml-20m-extended"
                                                                   ))