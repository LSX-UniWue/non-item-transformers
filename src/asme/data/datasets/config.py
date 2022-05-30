from pathlib import Path

from asme.core.init.context import Context
from asme.data.datamodule.config import DatasetPreprocessingConfig, PreprocessingConfigProvider
from asme.data.datamodule.converters import YooChooseConverter, Movielens1MConverter, ExampleConverter, \
    Movielens20MConverter, AmazonConverter, SteamConverter, SpotifyConverter, MelonConverter
from asme.data.datamodule.preprocessing.action import PREFIXES_KEY, DELIMITER_KEY, INPUT_DIR_KEY, OUTPUT_DIR_KEY
from asme.data.datamodule.preprocessing.csv import ConvertToCsv
from asme.data.datamodule.preprocessing.template import build_ratio_split, build_leave_one_out_split, \
    build_leave_percentage_out_split
from asme.data.datamodule.registry import register_preprocessing_config_provider
from asme.data.datamodule.unpacker import Unzipper
from asme.data.datasets.sequence import MetaInformation

"""
This file includes the definitions of all preprocessing steps that ASME should execute for any of the datasets included.

Generally, ASME will generate three types of splits for each dataset: 
- Ratio split:                      The data is split by session into three sets for training, validation and testing.
- Leave-One-Out (LOO) split:        The data is split intra-session, i.e. a session of n items is decomposed into a 
                                    n - 2 item training session, a 1 item validation and a 1 item test session. 
- Leave-Percentage-Out (LPO) split: Similar to the LOO split, the data is partioned intra-session. Specifically, the 
                                    first t% of each session is used for training, the next v% for validation and the 
                                    final (1 - t - v)% for testing. 

For each split, ASME will generate the necessary indices, vocabularies and popularities separately. The indices 
ususally include: 
- A session index which indicates the start byte and length of each session. In the case of a Ratio 
    split, three separate indices are generated for each split. 
- A next-item index which contains the start and end indices of every sub-session of each session as well as the
    corresponding target item index.
- A leave-one-out index which contains the start and end indices of every sub-session that is used for training as
    well as the corresponding target item index.
- A sliding-window index which can be used to train models using sliding windows of varying size. 

Each dataset preprocessing config can be customized by passing appropriate arguments to the functions via the config. 
These config parameters should have the same name as the function parameter and have to be placed into the 
`preprocessing` entry of the `datamodule` configuration entry.

Usually each preprocessing config has some dataset specific properties such as the extraction directory or output 
directory.
These are required and will result in a crash if not set via the config.
Additionally, each split can be configured independently via the following parameters (these are optional and have 
defaults given in the `register_preprocessing_config_provider` call after each preprocessing definition):

- Ratio split:
    - ratio_split_min_item_feedback:        The minimum number of interactions necessary for an item to be kept in the dataset.
    - ratio_split_min_sequence_length:      The minimum number of interactions in a session for it to be kept in the dataset.
    - ratio_split_train_percentage:         The fraction of session to (approximately) include into the training set.
    - ratio_split_validation_percentage:    The fraction of session to (approximately) include into the validation set.
    - ratio_split_test_percentage:          The fraction of session to (approximately) include into the test set.
    - ratio_split_window_markov_length:     The size of the sliding window that is used to extract samples from each session.
    - ratio_split_window_target_length:     The size of the sliding window that is used to extract targets from each session.
    - ratio_split_session_end_offset:       The distance between the session end and the last position for the sliding window.
    
- Leave-One-Out split:
    - loo_split_min_item_feedback:      The minimum number of interactions necessary for an item to be kept in the dataset.
    - loo_split_min_sequence_length:    The minimum number of interactions in a session for it to be kept in the dataset.
    
- Leave-Percentage-Out split:
    - lpo_split_min_item_feedback:      The minimum number of interactions necessary for an item to be kept in the dataset.
    - lpo_split_min_sequence_length:    The minimum number of interactions in a session for it to be kept in the dataset. 
    - lpo_split_train_percentage:       The fraction of session to (approximately) include into the training set. 
    - lpo_split_validation_percentage:  The fraction of session to (approximately) include into the validation set. 
    - lpo_split_test_percentage:        The fraction of session to (approximately) include into the test set. 
    - lpo_split_min_train_length:       The minimum size of each session in the training set. 
    - lpo_split_min_validation_length:  The minimum size of each session in the validation set. 
    - lpo_split_min_test_length:        The minimum size of each session in the test set. 

In order to register a preprocessing configuration provider yourself, you can use the 
`register_preprocessing_config_provider` function. When called, ASME will register your provider using the name and 
default values you provided. 
You can then use your provider by simply specifying its name as in the `dataset` parameter of the `datamodule` entry
in the configuration file.
"""


def get_ml_1m_preprocessing_config(
        # General parameters
        output_directory: str,
        extraction_directory: str,
        # Ratio split parameters
        ratio_split_min_item_feedback: int,
        ratio_split_min_sequence_length: int,
        ratio_split_train_percentage: float,
        ratio_split_validation_percentage: float,
        ratio_split_test_percentage: float,
        ratio_split_window_markov_length: int,
        ratio_split_window_target_length: int,
        ratio_split_session_end_offset: int,
        # Leave one out split parameters
        loo_split_min_item_feedback: int,
        loo_split_min_sequence_length: int,
        # Leave percentage out split parameters
        lpo_split_min_item_feedback: int,
        lpo_split_min_sequence_length: int,
        lpo_split_train_percentage: float,
        lpo_split_validation_percentage: float,
        lpo_split_test_percentage: float,
        lpo_split_min_train_length: int,
        lpo_split_min_validation_length: int,
        lpo_split_min_test_length: int,
) -> DatasetPreprocessingConfig:
    prefix = "ml-1m"
    context = Context()
    context.set(PREFIXES_KEY, [prefix])
    context.set(DELIMITER_KEY, "\t")
    context.set(INPUT_DIR_KEY, Path(extraction_directory))
    context.set(OUTPUT_DIR_KEY, Path(output_directory))

    columns = [MetaInformation("rating", type="int", run_tokenization=False),
               MetaInformation("gender", type="str"),
               MetaInformation("age", type="int", run_tokenization=False),
               MetaInformation("occupation", type="str"),
               MetaInformation("zip", type="str"),
               MetaInformation("title", type="str"),
               MetaInformation("year", type="str", run_tokenization=False),
               MetaInformation("genres", type="str", configs={"delimiter": "|"})]

    item_column = MetaInformation("item", column_name="title", type="str")
    min_item_feedback_column = "movieId"
    min_sequence_length_column = "userId"
    session_key = ["userId"]

    ratio_split_action = build_ratio_split(columns, prefix, ratio_split_min_item_feedback, min_item_feedback_column,
                                           ratio_split_min_sequence_length, min_sequence_length_column,
                                           session_key, [item_column], ratio_split_train_percentage,
                                           ratio_split_validation_percentage, ratio_split_test_percentage,
                                           ratio_split_window_markov_length, ratio_split_window_target_length,
                                           ratio_split_session_end_offset)

    leave_one_out_split_action = build_leave_one_out_split(columns, prefix, loo_split_min_item_feedback,
                                                           min_item_feedback_column,
                                                           loo_split_min_sequence_length, min_sequence_length_column,
                                                           session_key, item_column, [item_column])

    leave_percentage_out_split_action = build_leave_percentage_out_split(columns, prefix, lpo_split_min_item_feedback,
                                                                         min_item_feedback_column,
                                                                         lpo_split_min_sequence_length,
                                                                         min_sequence_length_column,
                                                                         session_key, item_column, [item_column],
                                                                         lpo_split_train_percentage,
                                                                         lpo_split_validation_percentage,
                                                                         lpo_split_test_percentage,
                                                                         lpo_split_min_train_length,
                                                                         lpo_split_min_validation_length,
                                                                         lpo_split_min_test_length)

    preprocessing_actions = [ConvertToCsv(Movielens1MConverter()),
                             ratio_split_action, leave_one_out_split_action,
                             leave_percentage_out_split_action]

    return DatasetPreprocessingConfig(prefix,
                                      "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
                                      Path(output_directory),
                                      Unzipper(Path(extraction_directory)),
                                      preprocessing_actions,
                                      context)


register_preprocessing_config_provider("ml-1m",
                                       PreprocessingConfigProvider(get_ml_1m_preprocessing_config,
                                                                   output_directory="./ml-1m",
                                                                   extraction_directory="./tmp/ml-1m",
                                                                   ratio_split_min_item_feedback=4,
                                                                   ratio_split_min_sequence_length=4,
                                                                   ratio_split_train_percentage=0.8,
                                                                   ratio_split_validation_percentage=0.1,
                                                                   ratio_split_test_percentage=0.1,
                                                                   ratio_split_window_markov_length=3,
                                                                   ratio_split_window_target_length=3,
                                                                   ratio_split_session_end_offset=0,
                                                                   # Leave one out split parameters
                                                                   loo_split_min_item_feedback=4,
                                                                   loo_split_min_sequence_length=4,
                                                                   # Leave percentage out split parameters
                                                                   lpo_split_min_item_feedback=4,
                                                                   lpo_split_min_sequence_length=4,
                                                                   lpo_split_train_percentage=0.8,
                                                                   lpo_split_validation_percentage=0.1,
                                                                   lpo_split_test_percentage=0.1,
                                                                   lpo_split_min_train_length=2,
                                                                   lpo_split_min_validation_length=1,
                                                                   lpo_split_min_test_length=1
                                                                   ))


def get_ml_20m_preprocessing_config(
        # General parameters
        output_directory: str,
        extraction_directory: str,
        # Ratio split parameters
        ratio_split_min_item_feedback: int,
        ratio_split_min_sequence_length: int,
        ratio_split_train_percentage: float,
        ratio_split_validation_percentage: float,
        ratio_split_test_percentage: float,
        ratio_split_window_markov_length: int,
        ratio_split_window_target_length: int,
        ratio_split_session_end_offset: int,
        # Leave one out split parameters
        loo_split_min_item_feedback: int,
        loo_split_min_sequence_length: int,
        # Leave percentage out split parameters
        lpo_split_min_item_feedback: int,
        lpo_split_min_sequence_length: int,
        lpo_split_train_percentage: float,
        lpo_split_validation_percentage: float,
        lpo_split_test_percentage: float,
        lpo_split_min_train_length: int,
        lpo_split_min_validation_length: int,
        lpo_split_min_test_length: int,
) -> DatasetPreprocessingConfig:
    prefix = "ml-20m"
    context = Context()
    context.set(PREFIXES_KEY, [prefix])
    context.set(DELIMITER_KEY, "\t")
    context.set(INPUT_DIR_KEY, Path(extraction_directory))
    context.set(OUTPUT_DIR_KEY, Path(output_directory))

    columns = [MetaInformation("rating", type="int", run_tokenization=False),
               MetaInformation("timestamp", type="str"),
               MetaInformation("title", type="str"),
               MetaInformation("genres", type="str", configs={"delimiter": "|"})]

    item_column = MetaInformation("item", column_name="title", type="str")
    min_item_feedback_column = "movieId"
    min_sequence_length_column = "userId"
    session_key = ["userId"]

    ratio_split_action = build_ratio_split(columns, prefix, ratio_split_min_item_feedback, min_item_feedback_column,
                                           ratio_split_min_sequence_length, min_sequence_length_column,
                                           session_key, [item_column], ratio_split_train_percentage,
                                           ratio_split_validation_percentage, ratio_split_test_percentage,
                                           ratio_split_window_markov_length, ratio_split_window_target_length,
                                           ratio_split_session_end_offset)

    leave_one_out_split_action = build_leave_one_out_split(columns, prefix, loo_split_min_item_feedback,
                                                           min_item_feedback_column,
                                                           loo_split_min_sequence_length, min_sequence_length_column,
                                                           session_key, item_column, [item_column])

    leave_percentage_out_split_action = build_leave_percentage_out_split(columns, prefix, lpo_split_min_item_feedback,
                                                                         min_item_feedback_column,
                                                                         lpo_split_min_sequence_length,
                                                                         min_sequence_length_column,
                                                                         session_key, item_column, [item_column],
                                                                         lpo_split_train_percentage,
                                                                         lpo_split_validation_percentage,
                                                                         lpo_split_test_percentage,
                                                                         lpo_split_min_train_length,
                                                                         lpo_split_min_validation_length,
                                                                         lpo_split_min_test_length)

    preprocessing_actions = [ConvertToCsv(Movielens20MConverter()),
                             ratio_split_action, leave_one_out_split_action,
                             leave_percentage_out_split_action]
    return DatasetPreprocessingConfig(prefix,
                                      "http://files.grouplens.org/datasets/movielens/ml-20m.zip",
                                      Path(output_directory),
                                      Unzipper(Path(extraction_directory)),
                                      preprocessing_actions,
                                      context)


register_preprocessing_config_provider("ml-20m",
                                       PreprocessingConfigProvider(get_ml_20m_preprocessing_config,
                                                                   output_directory="./ml-20m",
                                                                   extraction_directory="./tmp/ml-20m",
                                                                   ratio_split_min_item_feedback=4,
                                                                   ratio_split_min_sequence_length=4,
                                                                   ratio_split_train_percentage=0.8,
                                                                   ratio_split_validation_percentage=0.1,
                                                                   ratio_split_test_percentage=0.1,
                                                                   ratio_split_window_markov_length=3,
                                                                   ratio_split_window_target_length=3,
                                                                   ratio_split_session_end_offset=0,
                                                                   # Leave one out split parameters
                                                                   loo_split_min_item_feedback=4,
                                                                   loo_split_min_sequence_length=4,
                                                                   # Leave percentage out split parameters
                                                                   lpo_split_min_item_feedback=4,
                                                                   lpo_split_min_sequence_length=4,
                                                                   lpo_split_train_percentage=0.8,
                                                                   lpo_split_validation_percentage=0.1,
                                                                   lpo_split_test_percentage=0.1,
                                                                   lpo_split_min_train_length=2,
                                                                   lpo_split_min_validation_length=1,
                                                                   lpo_split_min_test_length=1
                                                                   ))


def get_amazon_preprocessing_config(
        # General parameters
        prefix: str,
        output_directory: str,
        input_directory: str,
        # Ratio split parameters
        ratio_split_min_item_feedback: int,
        ratio_split_min_sequence_length: int,
        ratio_split_train_percentage: float,
        ratio_split_validation_percentage: float,
        ratio_split_test_percentage: float,
        ratio_split_window_markov_length: int,
        ratio_split_window_target_length: int,
        ratio_split_session_end_offset: int,
        # Leave one out split parameters
        loo_split_min_item_feedback: int,
        loo_split_min_sequence_length: int,
        # Leave percentage out split parameters
        lpo_split_min_item_feedback: int,
        lpo_split_min_sequence_length: int,
        lpo_split_train_percentage: float,
        lpo_split_validation_percentage: float,
        lpo_split_test_percentage: float,
        lpo_split_min_train_length: int,
        lpo_split_min_validation_length: int,
        lpo_split_min_test_length: int,
) -> DatasetPreprocessingConfig:
    if prefix not in ["games", "beauty"]:
        raise KeyError("The only amazon datasets that are currently supported are 'games' and 'beauty'.")

    AMAZON_DOWNLOAD_URL_MAP = {
        "games": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games.json.gz",
        "beauty": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty.json.gz"
    }

    AMAZON_ZIPPED_FILE_NAMES = {
        "games": "reviews_Video_Games.json.gz",
        "beauty": "reviews_Beauty.json.gz"
    }

    context = Context()
    context.set(PREFIXES_KEY, [prefix])
    context.set(DELIMITER_KEY, "\t")
    context.set(INPUT_DIR_KEY, Path(input_directory) / AMAZON_ZIPPED_FILE_NAMES[prefix])
    context.set(OUTPUT_DIR_KEY, Path(output_directory))

    columns = [MetaInformation("reviewer_id", type="str"),
               MetaInformation("product_id", type="str"),
               MetaInformation("timestamp", type="int", run_tokenization=False)]

    min_item_feedback_column = "product_id"
    min_sequence_length_column = "reviewer_id"
    session_key = ["reviewer_id"]
    item_column = MetaInformation("item", column_name="product_id", type="str")

    ratio_split_action = build_ratio_split(columns, prefix, ratio_split_min_item_feedback, min_item_feedback_column,
                                           ratio_split_min_sequence_length, min_sequence_length_column,
                                           session_key, [item_column], ratio_split_train_percentage,
                                           ratio_split_validation_percentage, ratio_split_test_percentage,
                                           ratio_split_window_markov_length, ratio_split_window_target_length,
                                           ratio_split_session_end_offset)

    leave_one_out_split_action = build_leave_one_out_split(columns, prefix, loo_split_min_item_feedback,
                                                           min_item_feedback_column,
                                                           loo_split_min_sequence_length, min_sequence_length_column,
                                                           session_key, item_column, [item_column])

    leave_percentage_out_split_action = build_leave_percentage_out_split(columns, prefix, lpo_split_min_item_feedback,
                                                                         min_item_feedback_column,
                                                                         lpo_split_min_sequence_length,
                                                                         min_sequence_length_column,
                                                                         session_key, item_column, [item_column],
                                                                         lpo_split_train_percentage,
                                                                         lpo_split_validation_percentage,
                                                                         lpo_split_test_percentage,
                                                                         lpo_split_min_train_length,
                                                                         lpo_split_min_validation_length,
                                                                         lpo_split_min_test_length)

    preprocessing_actions = [ConvertToCsv(AmazonConverter()),
                             ratio_split_action, leave_one_out_split_action,
                             leave_percentage_out_split_action]

    return DatasetPreprocessingConfig(prefix,
                                      AMAZON_DOWNLOAD_URL_MAP[prefix],
                                      Path(output_directory),
                                      None,
                                      preprocessing_actions,
                                      context)


register_preprocessing_config_provider("beauty",
                                       PreprocessingConfigProvider(get_amazon_preprocessing_config,
                                                                   prefix="beauty",
                                                                   output_directory="./beauty",
                                                                   ratio_split_min_item_feedback=4,
                                                                   ratio_split_min_sequence_length=4,
                                                                   ratio_split_train_percentage=0.8,
                                                                   ratio_split_validation_percentage=0.1,
                                                                   ratio_split_test_percentage=0.1,
                                                                   ratio_split_window_markov_length=3,
                                                                   ratio_split_window_target_length=3,
                                                                   ratio_split_session_end_offset=0,
                                                                   # Leave one out split parameters
                                                                   loo_split_min_item_feedback=4,
                                                                   loo_split_min_sequence_length=4,
                                                                   # Leave percentage out split parameters
                                                                   lpo_split_min_item_feedback=4,
                                                                   lpo_split_min_sequence_length=4,
                                                                   lpo_split_train_percentage=0.8,
                                                                   lpo_split_validation_percentage=0.1,
                                                                   lpo_split_test_percentage=0.1,
                                                                   lpo_split_min_train_length=2,
                                                                   lpo_split_min_validation_length=1,
                                                                   lpo_split_min_test_length=1
                                                                   ))

register_preprocessing_config_provider("games",
                                       PreprocessingConfigProvider(get_amazon_preprocessing_config,
                                                                   prefix="games",
                                                                   output_directory="./games",
                                                                   ratio_split_min_item_feedback=4,
                                                                   ratio_split_min_sequence_length=4,
                                                                   ratio_split_train_percentage=0.8,
                                                                   ratio_split_validation_percentage=0.1,
                                                                   ratio_split_test_percentage=0.1,
                                                                   ratio_split_window_markov_length=3,
                                                                   ratio_split_window_target_length=3,
                                                                   ratio_split_session_end_offset=0,
                                                                   # Leave one out split parameters
                                                                   loo_split_min_item_feedback=4,
                                                                   loo_split_min_sequence_length=4,
                                                                   # Leave percentage out split parameters
                                                                   lpo_split_min_item_feedback=4,
                                                                   lpo_split_min_sequence_length=4,
                                                                   lpo_split_train_percentage=0.8,
                                                                   lpo_split_validation_percentage=0.1,
                                                                   lpo_split_test_percentage=0.1,
                                                                   lpo_split_min_train_length=2,
                                                                   lpo_split_min_validation_length=1,
                                                                   lpo_split_min_test_length=1))


def get_steam_preprocessing_config(
        # General parameters
        output_directory: str,
        input_dir: str,
        # Ratio split parameters
        ratio_split_min_item_feedback: int,
        ratio_split_min_sequence_length: int,
        ratio_split_train_percentage: float,
        ratio_split_validation_percentage: float,
        ratio_split_test_percentage: float,
        ratio_split_window_markov_length: int,
        ratio_split_window_target_length: int,
        ratio_split_session_end_offset: int,
        # Leave one out split parameters
        loo_split_min_item_feedback: int,
        loo_split_min_sequence_length: int,
        # Leave percentage out split parameters
        lpo_split_min_item_feedback: int,
        lpo_split_min_sequence_length: int,
        lpo_split_train_percentage: float,
        lpo_split_validation_percentage: float,
        lpo_split_test_percentage: float,
        lpo_split_min_train_length: int,
        lpo_split_min_validation_length: int,
        lpo_split_min_test_length: int
) -> DatasetPreprocessingConfig:
    prefix = "steam"
    filename = "steam_reviews.json.gz"

    context = Context()
    context.set(PREFIXES_KEY, [prefix])
    context.set(DELIMITER_KEY, "\t")
    context.set(INPUT_DIR_KEY, Path(input_dir) / filename)
    context.set(OUTPUT_DIR_KEY, Path(output_directory))

    columns = [MetaInformation("username", type="str"),
               MetaInformation("product_id", type="str"),
               MetaInformation("date", type="timestamp", configs={"format": "%Y-%m-%d"}, run_tokenization=False)]

    min_item_feedback_column = "product_id"
    min_sequence_length_column = "username"
    session_key = ["username"]
    item_column = MetaInformation("item", column_name="product_id", type="str")

    ratio_split_action = build_ratio_split(columns, prefix, ratio_split_min_item_feedback, min_item_feedback_column,
                                           ratio_split_min_sequence_length, min_sequence_length_column,
                                           session_key, [item_column], ratio_split_train_percentage,
                                           ratio_split_validation_percentage, ratio_split_test_percentage,
                                           ratio_split_window_markov_length, ratio_split_window_target_length,
                                           ratio_split_session_end_offset)

    leave_one_out_split_action = build_leave_one_out_split(columns, prefix, loo_split_min_item_feedback,
                                                           min_item_feedback_column,
                                                           loo_split_min_sequence_length, min_sequence_length_column,
                                                           session_key, item_column, [item_column])

    leave_percentage_out_split_action = build_leave_percentage_out_split(columns, prefix, lpo_split_min_item_feedback,
                                                                         min_item_feedback_column,
                                                                         lpo_split_min_sequence_length,
                                                                         min_sequence_length_column,
                                                                         session_key, item_column, [item_column],
                                                                         lpo_split_train_percentage,
                                                                         lpo_split_validation_percentage,
                                                                         lpo_split_test_percentage,
                                                                         lpo_split_min_train_length,
                                                                         lpo_split_min_validation_length,
                                                                         lpo_split_min_test_length)

    preprocessing_actions = [ConvertToCsv(SteamConverter()),
                             ratio_split_action, leave_one_out_split_action,
                             leave_percentage_out_split_action]

    return DatasetPreprocessingConfig(prefix,
                                      "http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz",
                                      Path(output_directory),
                                      None,
                                      preprocessing_actions,
                                      context)


register_preprocessing_config_provider("steam",
                                       PreprocessingConfigProvider(get_steam_preprocessing_config,
                                                                   output_directory="./steam",
                                                                   ratio_split_min_item_feedback=4,
                                                                   ratio_split_min_sequence_length=4,
                                                                   ratio_split_train_percentage=0.8,
                                                                   ratio_split_validation_percentage=0.1,
                                                                   ratio_split_test_percentage=0.1,
                                                                   ratio_split_window_markov_length=3,
                                                                   ratio_split_window_target_length=3,
                                                                   ratio_split_session_end_offset=0,
                                                                   # Leave one out split parameters
                                                                   loo_split_min_item_feedback=4,
                                                                   loo_split_min_sequence_length=4,
                                                                   # Leave percentage out split parameters
                                                                   lpo_split_min_item_feedback=4,
                                                                   lpo_split_min_sequence_length=4,
                                                                   lpo_split_train_percentage=0.8,
                                                                   lpo_split_validation_percentage=0.1,
                                                                   lpo_split_test_percentage=0.1,
                                                                   lpo_split_min_train_length=2,
                                                                   lpo_split_min_validation_length=1,
                                                                   lpo_split_min_test_length=1
                                                                   ))


def get_yoochoose_preprocessing_config(
        # General parameters
        output_directory: str,
        input_directory: str,
        # Ratio split parameters
        ratio_split_min_item_feedback: int,
        ratio_split_min_sequence_length: int,
        ratio_split_train_percentage: float,
        ratio_split_validation_percentage: float,
        ratio_split_test_percentage: float,
        ratio_split_window_markov_length: int,
        ratio_split_window_target_length: int,
        ratio_split_session_end_offset: int,
        # Leave one out split parameters
        loo_split_min_item_feedback: int,
        loo_split_min_sequence_length: int,
        # Leave percentage out split parameters
        lpo_split_min_item_feedback: int,
        lpo_split_min_sequence_length: int,
        lpo_split_train_percentage: float,
        lpo_split_validation_percentage: float,
        lpo_split_test_percentage: float,
        lpo_split_min_train_length: int,
        lpo_split_min_validation_length: int,
        lpo_split_min_test_length: int
) -> DatasetPreprocessingConfig:
    prefix = "yoochoose"
    context = Context()
    context.set(PREFIXES_KEY, [prefix])
    context.set(DELIMITER_KEY, "\t")
    context.set(INPUT_DIR_KEY, Path(input_directory))
    context.set(OUTPUT_DIR_KEY, Path(output_directory))

    columns = [MetaInformation("SessionId", type="str"),
               MetaInformation("ItemId", type="str"),
               MetaInformation("Time", type="int", run_tokenization=False)]

    min_item_feedback_column = "ItemId"
    min_sequence_length_column = "SessionId"
    session_key = ["SessionId"]
    item_column = MetaInformation("item", column_name="ItemId", type="str")

    ratio_split_action = build_ratio_split(columns, prefix, ratio_split_min_item_feedback, min_item_feedback_column,
                                           ratio_split_min_sequence_length, min_sequence_length_column,
                                           session_key, [item_column], ratio_split_train_percentage,
                                           ratio_split_validation_percentage, ratio_split_test_percentage,
                                           ratio_split_window_markov_length, ratio_split_window_target_length,
                                           ratio_split_session_end_offset)

    leave_one_out_split_action = build_leave_one_out_split(columns, prefix, loo_split_min_item_feedback,
                                                           min_item_feedback_column,
                                                           loo_split_min_sequence_length, min_sequence_length_column,
                                                           session_key, item_column, [item_column])

    leave_percentage_out_split_action = build_leave_percentage_out_split(columns, prefix, lpo_split_min_item_feedback,
                                                                         min_item_feedback_column,
                                                                         lpo_split_min_sequence_length,
                                                                         min_sequence_length_column,
                                                                         session_key, item_column, [item_column],
                                                                         lpo_split_train_percentage,
                                                                         lpo_split_validation_percentage,
                                                                         lpo_split_test_percentage,
                                                                         lpo_split_min_train_length,
                                                                         lpo_split_min_validation_length,
                                                                         lpo_split_min_test_length)

    preprocessing_actions = [ConvertToCsv(YooChooseConverter()),
                             ratio_split_action, leave_one_out_split_action,
                             leave_percentage_out_split_action]

    return DatasetPreprocessingConfig(prefix,
                                      None,
                                      Path(output_directory),
                                      None,
                                      preprocessing_actions,
                                      context)


register_preprocessing_config_provider("yoochoose",
                                       PreprocessingConfigProvider(get_yoochoose_preprocessing_config,
                                                                   output_directory="./yoochoose",
                                                                   ratio_split_min_item_feedback=4,
                                                                   ratio_split_min_sequence_length=4,
                                                                   ratio_split_train_percentage=0.8,
                                                                   ratio_split_validation_percentage=0.1,
                                                                   ratio_split_test_percentage=0.1,
                                                                   ratio_split_window_markov_length=3,
                                                                   ratio_split_window_target_length=3,
                                                                   ratio_split_session_end_offset=0,
                                                                   # Leave one out split parameters
                                                                   loo_split_min_item_feedback=4,
                                                                   loo_split_min_sequence_length=4,
                                                                   # Leave percentage out split parameters
                                                                   lpo_split_min_item_feedback=4,
                                                                   lpo_split_min_sequence_length=4,
                                                                   lpo_split_train_percentage=0.8,
                                                                   lpo_split_validation_percentage=0.1,
                                                                   lpo_split_test_percentage=0.1,
                                                                   lpo_split_min_train_length=2,
                                                                   lpo_split_min_validation_length=1,
                                                                   lpo_split_min_test_length=1))


def get_spotify_preprocessing_config(  # General parameters
        output_directory: str,
        input_directory: str,
        # Ratio split parameters
        ratio_split_min_item_feedback: int,
        ratio_split_min_sequence_length: int,
        ratio_split_train_percentage: float,
        ratio_split_validation_percentage: float,
        ratio_split_test_percentage: float,
        ratio_split_window_markov_length: int,
        ratio_split_window_target_length: int,
        ratio_split_session_end_offset: int,
        # Leave one out split parameters
        loo_split_min_item_feedback: int,
        loo_split_min_sequence_length: int,
        # Leave percentage out split parameters
        lpo_split_min_item_feedback: int,
        lpo_split_min_sequence_length: int,
        lpo_split_train_percentage: float,
        lpo_split_validation_percentage: float,
        lpo_split_test_percentage: float,
        lpo_split_min_train_length: int,
        lpo_split_min_validation_length: int,
        lpo_split_min_test_length: int
) -> DatasetPreprocessingConfig:
    prefix = "spotify"
    context = Context()
    context.set(PREFIXES_KEY, [prefix])
    context.set(DELIMITER_KEY, "\t")
    context.set(INPUT_DIR_KEY, Path(input_directory))
    context.set(OUTPUT_DIR_KEY, Path(output_directory))

    columns = [MetaInformation("playlist_id", type="str"),
               MetaInformation("playlist_timestamp", type="str", run_tokenization=False),
               MetaInformation("track_name", type="str"),
               MetaInformation("album_name", type="str"),
               MetaInformation("artist_name", type="str")]

    min_item_feedback_column = "track_name"
    min_sequence_length_column = "playlist_id"
    session_key = ["playlist_id"]
    item_column = MetaInformation("item", column_name="track_name", type="str")

    ratio_split_action = build_ratio_split(columns, prefix, ratio_split_min_item_feedback, min_item_feedback_column,
                                           ratio_split_min_sequence_length, min_sequence_length_column,
                                           session_key, [item_column], ratio_split_train_percentage,
                                           ratio_split_validation_percentage, ratio_split_test_percentage,
                                           ratio_split_window_markov_length, ratio_split_window_target_length,
                                           ratio_split_session_end_offset)

    leave_one_out_split_action = build_leave_one_out_split(columns, prefix, loo_split_min_item_feedback,
                                                           min_item_feedback_column,
                                                           loo_split_min_sequence_length, min_sequence_length_column,
                                                           session_key, item_column, [item_column])

    leave_percentage_out_split_action = build_leave_percentage_out_split(columns, prefix, lpo_split_min_item_feedback,
                                                                         min_item_feedback_column,
                                                                         lpo_split_min_sequence_length,
                                                                         min_sequence_length_column,
                                                                         session_key, item_column, [item_column],
                                                                         lpo_split_train_percentage,
                                                                         lpo_split_validation_percentage,
                                                                         lpo_split_test_percentage,
                                                                         lpo_split_min_train_length,
                                                                         lpo_split_min_validation_length,
                                                                         lpo_split_min_test_length)

    preprocessing_actions = [ConvertToCsv(SpotifyConverter()),
                             ratio_split_action, leave_one_out_split_action,
                             leave_percentage_out_split_action]

    return DatasetPreprocessingConfig(prefix,
                                      None,
                                      Path(output_directory),
                                      None,
                                      preprocessing_actions,
                                      context)


register_preprocessing_config_provider("spotify",
                                       PreprocessingConfigProvider(get_spotify_preprocessing_config,
                                                                   output_directory="./spotify",
                                                                   ratio_split_min_item_feedback=4,
                                                                   ratio_split_min_sequence_length=4,
                                                                   ratio_split_train_percentage=0.8,
                                                                   ratio_split_validation_percentage=0.1,
                                                                   ratio_split_test_percentage=0.1,
                                                                   ratio_split_window_markov_length=3,
                                                                   ratio_split_window_target_length=3,
                                                                   ratio_split_session_end_offset=0,
                                                                   # Leave one out split parameters
                                                                   loo_split_min_item_feedback=4,
                                                                   loo_split_min_sequence_length=4,
                                                                   # Leave percentage out split parameters
                                                                   lpo_split_min_item_feedback=4,
                                                                   lpo_split_min_sequence_length=4,
                                                                   lpo_split_train_percentage=0.8,
                                                                   lpo_split_validation_percentage=0.1,
                                                                   lpo_split_test_percentage=0.1,
                                                                   lpo_split_min_train_length=2,
                                                                   lpo_split_min_validation_length=1,
                                                                   lpo_split_min_test_length=1))


def get_melon_preprocessing_config(
        # General parameters
        output_directory: str,
        input_directory: str,
        # Ratio split parameters
        ratio_split_min_item_feedback: int,
        ratio_split_min_sequence_length: int,
        ratio_split_train_percentage: float,
        ratio_split_validation_percentage: float,
        ratio_split_test_percentage: float,
        ratio_split_window_markov_length: int,
        ratio_split_window_target_length: int,
        ratio_split_session_end_offset: int,
        # Leave one out split parameters
        loo_split_min_item_feedback: int,
        loo_split_min_sequence_length: int,
        # Leave percentage out split parameters
        lpo_split_min_item_feedback: int,
        lpo_split_min_sequence_length: int,
        lpo_split_train_percentage: float,
        lpo_split_validation_percentage: float,
        lpo_split_test_percentage: float,
        lpo_split_min_train_length: int,
        lpo_split_min_validation_length: int,
        lpo_split_min_test_length: int
) -> DatasetPreprocessingConfig:
    prefix = "melon"
    context = Context()
    context.set(PREFIXES_KEY, [prefix])
    context.set(DELIMITER_KEY, "\t")
    context.set(INPUT_DIR_KEY, Path(input_directory))
    context.set(OUTPUT_DIR_KEY, Path(output_directory))

    columns = [MetaInformation("playlist_id", type="str"),
               MetaInformation("track_name", type="str"),
               MetaInformation("album_name", type="str"),
               MetaInformation("artist_name", type="str", configs={
                   "element_type": "str",
                   "delimiter": "|"
               }),
               MetaInformation("genre", type="list", configs={
                   "element_type": "str",
                   "delimiter": "|"})
               ]

    min_item_feedback_column = "track_name"
    min_sequence_length_column = "playlist_id"
    session_key = ["playlist_id"]
    item_column = MetaInformation("item", column_name="track_name", type="str")

    ratio_split_action = build_ratio_split(columns, prefix, ratio_split_min_item_feedback, min_item_feedback_column,
                                           ratio_split_min_sequence_length, min_sequence_length_column,
                                           session_key, [item_column], ratio_split_train_percentage,
                                           ratio_split_validation_percentage, ratio_split_test_percentage,
                                           ratio_split_window_markov_length, ratio_split_window_target_length,
                                           ratio_split_session_end_offset)

    leave_one_out_split_action = build_leave_one_out_split(columns, prefix, loo_split_min_item_feedback,
                                                           min_item_feedback_column,
                                                           loo_split_min_sequence_length, min_sequence_length_column,
                                                           session_key, item_column, [item_column])

    leave_percentage_out_split_action = build_leave_percentage_out_split(columns, prefix, lpo_split_min_item_feedback,
                                                                         min_item_feedback_column,
                                                                         lpo_split_min_sequence_length,
                                                                         min_sequence_length_column,
                                                                         session_key, item_column, [item_column],
                                                                         lpo_split_train_percentage,
                                                                         lpo_split_validation_percentage,
                                                                         lpo_split_test_percentage,
                                                                         lpo_split_min_train_length,
                                                                         lpo_split_min_validation_length,
                                                                         lpo_split_min_test_length)

    preprocessing_actions = [ConvertToCsv(MelonConverter()),
                             ratio_split_action,
                             leave_one_out_split_action,
                             leave_percentage_out_split_action
                             ]

    return DatasetPreprocessingConfig(prefix,
                                      None,
                                      Path(output_directory),
                                      None,
                                      preprocessing_actions,
                                      context)


register_preprocessing_config_provider("melon",
                                       PreprocessingConfigProvider(get_melon_preprocessing_config,
                                                                   output_directory="./melon",
                                                                   ratio_split_min_item_feedback=4,
                                                                   ratio_split_min_sequence_length=4,
                                                                   ratio_split_train_percentage=0.8,
                                                                   ratio_split_validation_percentage=0.1,
                                                                   ratio_split_test_percentage=0.1,
                                                                   ratio_split_window_markov_length=3,
                                                                   ratio_split_window_target_length=3,
                                                                   ratio_split_session_end_offset=0,
                                                                   # Leave one out split parameters
                                                                   loo_split_min_item_feedback=4,
                                                                   loo_split_min_sequence_length=4,
                                                                   # Leave percentage out split parameters
                                                                   lpo_split_min_item_feedback=4,
                                                                   lpo_split_min_sequence_length=4,
                                                                   lpo_split_train_percentage=0.8,
                                                                   lpo_split_validation_percentage=0.1,
                                                                   lpo_split_test_percentage=0.1,
                                                                   lpo_split_min_train_length=2,
                                                                   lpo_split_min_validation_length=1,
                                                                   lpo_split_min_test_length=1))

"""
hier müssen wir noch die Parameter ändern:
window_size setzt sich aus sequence length und target_length zusammen
min_input_length bei extractor??
außerdem bei LOO auch Sliding_window index??
"""


def get_example_preprocessing_config(
        # General parameters
        output_directory: str,
        input_file_path: str,
        # Ratio split parameters
        ratio_split_min_item_feedback: int,
        ratio_split_min_sequence_length: int,
        ratio_split_train_percentage: float,
        ratio_split_validation_percentage: float,
        ratio_split_test_percentage: float,
        ratio_split_window_markov_length: int,
        ratio_split_window_target_length: int,
        ratio_split_session_end_offset: int,
        # Leave one out split parameters
        loo_split_min_item_feedback: int,
        loo_split_min_sequence_length: int,
        # Leave percentage out split parameters
        lpo_split_min_item_feedback: int,
        lpo_split_min_sequence_length: int,
        lpo_split_train_percentage: float,
        lpo_split_validation_percentage: float,
        lpo_split_test_percentage: float,
        lpo_split_min_train_length: int,
        lpo_split_min_validation_length: int,
        lpo_split_min_test_length: int
) -> DatasetPreprocessingConfig:
    prefix = "example"
    context = Context()
    context.set(PREFIXES_KEY, [prefix])
    context.set(DELIMITER_KEY, "\t")
    context.set(OUTPUT_DIR_KEY, Path(output_directory))
    context.set(INPUT_DIR_KEY, input_file_path)

    # FIXME (AD) we're forced to set column_name because vocabulary and popularity code relies on it being set.
    columns = [MetaInformation("item_id", column_name="item_id", type="str"),
               # TODO (AD) find out why setting type to int prevents correct vocabulary creation (vocabulary is not saved with consecutive ids)
               MetaInformation("user_id", column_name="user_id", type="str"),
               MetaInformation("session_id", column_name="session_id", type="str"),
               MetaInformation("attr_one", column_name="attr_one", type="str")]

    min_item_feedback_column = "item_id"
    min_sequence_length_column = "session_id"
    session_key = ["session_id"]
    item_column = MetaInformation("item", column_name="item_id", type="str")

    ratio_split_action = build_ratio_split(columns, prefix, ratio_split_min_item_feedback, min_item_feedback_column,
                                           ratio_split_min_sequence_length, min_sequence_length_column,
                                           session_key, [item_column], ratio_split_train_percentage,
                                           ratio_split_validation_percentage, ratio_split_test_percentage,
                                           ratio_split_window_markov_length, ratio_split_window_target_length,
                                           ratio_split_session_end_offset)

    leave_one_out_split_action = build_leave_one_out_split(columns, prefix, loo_split_min_item_feedback,
                                                           min_item_feedback_column,
                                                           loo_split_min_sequence_length, min_sequence_length_column,
                                                           session_key, item_column, [item_column])

    leave_percentage_out_split_action = build_leave_percentage_out_split(columns, prefix, lpo_split_min_item_feedback,
                                                                         min_item_feedback_column,
                                                                         lpo_split_min_sequence_length,
                                                                         min_sequence_length_column,
                                                                         session_key, item_column, [item_column],
                                                                         lpo_split_train_percentage,
                                                                         lpo_split_validation_percentage,
                                                                         lpo_split_test_percentage,
                                                                         lpo_split_min_train_length,
                                                                         lpo_split_min_validation_length,
                                                                         lpo_split_min_test_length)

    preprocessing_actions = [ConvertToCsv(ExampleConverter()),
                             ratio_split_action, leave_one_out_split_action,
                             leave_percentage_out_split_action]

    return DatasetPreprocessingConfig(prefix,
                                      None,
                                      Path(output_directory),
                                      None,
                                      preprocessing_actions,
                                      context)


register_preprocessing_config_provider("example",
                                       PreprocessingConfigProvider(get_example_preprocessing_config,
                                                                   output_directory="./example",
                                                                   input_file_path="../tests/example_dataset/example.csv",
                                                                   ratio_split_min_item_feedback=0,
                                                                   ratio_split_min_sequence_length=3,
                                                                   ratio_split_train_percentage=0.7,
                                                                   ratio_split_validation_percentage=0.1,
                                                                   ratio_split_test_percentage=0.2,
                                                                   ratio_split_window_markov_length=2,
                                                                   ratio_split_window_target_length=1,
                                                                   ratio_split_session_end_offset=0,
                                                                   # Leave one out split parameters
                                                                   loo_split_min_item_feedback=0,
                                                                   loo_split_min_sequence_length=3,
                                                                   # Leave percentage out split parameters
                                                                   lpo_split_min_item_feedback=4,
                                                                   lpo_split_min_sequence_length=4,
                                                                   lpo_split_train_percentage=0.8,
                                                                   lpo_split_validation_percentage=0.1,
                                                                   lpo_split_test_percentage=0.1,
                                                                   lpo_split_min_train_length=2,
                                                                   lpo_split_min_validation_length=1,
                                                                   lpo_split_min_test_length=1
                                                                   ))
