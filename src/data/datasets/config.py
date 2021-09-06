from pathlib import Path
from typing import Optional

from asme.init.context import Context
from data.datamodule.config import DatasetPreprocessingConfig, register_preprocessing_config_provider, \
    PreprocessingConfigProvider
from data.datamodule.preprocessing import ConvertToCsv, TransformCsv, CreateSessionIndex, \
    GroupAndFilter, GroupedFilter, CreateVocabulary, DELIMITER_KEY, \
    OUTPUT_DIR_KEY, CreateRatioSplit, CreateNextItemIndex, CreateLeaveOneOutSplit, CreatePopularity, \
    MAIN_FILE_KEY, UseExistingCsv, UseExistingSplit, SPLIT_BASE_DIRECTORY_PATH, \
    CreateSlidingWindowIndex, INPUT_DIR_KEY, SESSION_INDEX_KEY, CopyMainFile
from data.datamodule.converters import YooChooseConverter, Movielens1MConverter, ExampleConverter, \
    Movielens20MConverter, AmazonConverter, SteamConverter
from data.datamodule.extractors import RemainingSessionPositionExtractor, SlidingWindowPositionExtractor
from data.datamodule.unpacker import Unzipper
from data.datamodule.preprocessing import PREFIXES_KEY
from data.datasets.sequence import MetaInformation


def get_ml_1m_preprocessing_config(output_directory: str,
                                   extraction_directory: str,
                                   min_item_feedback: int,
                                   min_sequence_length: int,
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
               MetaInformation("genres", type="str", configs={"delimiter": "|"})]

    preprocessing_actions = [ConvertToCsv(Movielens1MConverter()),
                             GroupAndFilter("items_filtered", "movieId", GroupedFilter("count", lambda v: v >= min_item_feedback)),
                             GroupAndFilter("sessions_filtered", "userId", GroupedFilter("count", lambda v: v >= min_sequence_length)),
                             CreateSessionIndex(["userId"]),
                             CreateRatioSplit(0.8, 0.1, 0.1,
                                              per_split_actions=
                                                  [CreateSessionIndex(["userId"]),
                                                   CreateNextItemIndex(
                                                       [MetaInformation("item", column_name="title", type="str")],
                                                       RemainingSessionPositionExtractor(
                                                           min_sequence_length))],
                                              complete_split_actions=
                                                  [CreateVocabulary(columns, prefixes=[prefix]),
                                                   CreatePopularity(columns, prefixes=[prefix])]),
                             CreateLeaveOneOutSplit(MetaInformation("item", column_name="title", type="str"),
                                                    inner_actions=
                                                     [CreateNextItemIndex(
                                                        [MetaInformation("item", column_name="title", type="str")],
                                                        RemainingSessionPositionExtractor(
                                                            min_sequence_length)),
                                                      CreateVocabulary(columns),
                                                      CreatePopularity(columns)]),
                             CopyMainFile()
                             ]
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
                                                                   min_item_feedback=4,
                                                                   min_sequence_length=4))


def get_ml_20m_preprocessing_config(output_directory: str,
                                    extraction_directory: str,
                                    min_item_feedback: int,
                                    min_sequence_length: int
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

    preprocessing_actions = [ConvertToCsv(Movielens20MConverter()),
                             GroupAndFilter("items_filtered", "movieId", GroupedFilter("count", lambda v: v >= min_item_feedback)),
                             # We can drop the movieId column after filtering
                             TransformCsv("movieId_dropped",lambda df: df.drop("movieId", axis=1)),
                             GroupAndFilter("sessions_filtered","userId", GroupedFilter("count", lambda v: v >= min_sequence_length)),
                             CreateSessionIndex(["userId"]),
                             CreateRatioSplit(0.8, 0.1, 0.1,
                                              per_split_actions=
                                              [CreateSessionIndex(["userId"]),
                                               CreateNextItemIndex(
                                                   [MetaInformation("item", column_name="title", type="str")],
                                                   RemainingSessionPositionExtractor(
                                                       min_sequence_length))],
                                              complete_split_actions=
                                              [CreateVocabulary(columns, prefixes=[prefix]),
                                               CreatePopularity(columns, prefixes=[prefix])]),
                             CreateLeaveOneOutSplit(MetaInformation("item", column_name="title", type="str"),
                                                    inner_actions=
                                                    [CreateNextItemIndex(
                                                        [MetaInformation("item", column_name="title", type="str")],
                                                        RemainingSessionPositionExtractor(
                                                            min_sequence_length)),
                                                        CreateVocabulary(columns),
                                                        CreatePopularity(columns)]),
                             CopyMainFile()
                             ]
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
                                                                   min_item_feedback=4,
                                                                   min_sequence_length=4))


def get_amazon_preprocessing_config(prefix: str,
                                    output_directory: str,
                                    input_directory: str,
                                    min_item_feedback=5,
                                    min_sequence_length=5
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

    preprocessing_actions = [ConvertToCsv(AmazonConverter()),
                             GroupAndFilter("items_filtered", "product_id", GroupedFilter("count", lambda v: v >= min_item_feedback)),
                             GroupAndFilter("sessions_filtered", "reviewer_id", GroupedFilter("count", lambda v: v >= min_sequence_length)),
                             CreateSessionIndex(["reviewer_id"]),
                             CreateRatioSplit(0.8, 0.1, 0.1,
                                              per_split_actions=
                                              [CreateSessionIndex(["reviewer_id"]),
                                               CreateNextItemIndex(
                                                   [MetaInformation("item", column_name="product_id", type="str")],
                                                   RemainingSessionPositionExtractor(
                                                       min_sequence_length))],
                                              complete_split_actions=
                                              [CreateVocabulary(columns, prefixes=[prefix]),
                                               CreatePopularity(columns, prefixes=[prefix])]),
                             CreateLeaveOneOutSplit(MetaInformation("item", column_name="product_id", type="str"),
                                                    inner_actions=
                                                    [CreateNextItemIndex(
                                                        [MetaInformation("item", column_name="product_id", type="str")],
                                                        RemainingSessionPositionExtractor(
                                                            min_sequence_length)),
                                                        CreateVocabulary(columns),
                                                        CreatePopularity(columns)]),
                             CopyMainFile()
                             ]

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
                                                                   min_item_feedback=4,
                                                                   min_sequence_length=4))

register_preprocessing_config_provider("games",
                                       PreprocessingConfigProvider(get_amazon_preprocessing_config,
                                                                   prefix="games",
                                                                   output_directory="./games",
                                                                   min_item_feedback=4,
                                                                   min_sequence_length=4))


def get_dota_shop_preprocessing_config(output_directory: str,
                                       raw_csv_file_path: str,
                                       split_directory: str,
                                       min_sequence_length: int,
                                       window_size: Optional[int] = None,
                                       session_end_offset: Optional[int] = None) -> DatasetPreprocessingConfig:
    prefix = "dota"
    context = Context()
    context.set(PREFIXES_KEY, [prefix])
    context.set(DELIMITER_KEY, "\t")
    context.set(OUTPUT_DIR_KEY, Path(output_directory))

    full_csv_file_path = Path(raw_csv_file_path)
    context.set(INPUT_DIR_KEY, full_csv_file_path)

    # assume that the split directory is relative to the full csv file.
    split_directory_path = full_csv_file_path.parent / split_directory
    context.set(SPLIT_BASE_DIRECTORY_PATH, split_directory_path)

    columns = [MetaInformation("hero_id", column_name="hero_id", type="str"),
               MetaInformation("item_id", column_name="item_id", type="str")]

    preprocessing_actions = [UseExistingCsv(),
                             CreateSessionIndex(["id", "hero_id"]),
                             CreateVocabulary(columns, prefixes=[prefix]),
                             UseExistingSplit(
                                 split_names=["train", "validation", "test"],
                                 per_split_actions=
                                 [
                                     CreateSessionIndex(["id", "hero_id"]),
                                     CreateNextItemIndex(
                                         [MetaInformation("item", column_name="item_id", type="str")],
                                         RemainingSessionPositionExtractor(min_sequence_length)
                                     ),
                                     CreateSlidingWindowIndex(
                                         [MetaInformation("item", column_name="item_id", type="str")],
                                         SlidingWindowPositionExtractor(window_size, session_end_offset)
                                     )
                                 ])
                             ]

    return DatasetPreprocessingConfig(prefix,
                                      None,
                                      Path(output_directory),
                                      None,
                                      preprocessing_actions,
                                      context)


register_preprocessing_config_provider("dota",
                                       PreprocessingConfigProvider(get_dota_shop_preprocessing_config,
                                                                   output_directory="./dota",
                                                                   raw_csv_file_path="./dota",
                                                                   split_directory="split-0.8_0.1_0.1",
                                                                   min_sequence_length=2,
                                                                   window_size=12,
                                                                   session_end_offset=0))


def get_steam_preprocessing_config(output_directory: str,
                                   input_dir: str,
                                   min_item_feedback=5,
                                   min_sequence_length=5
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
               MetaInformation("date", type="timestamp",configs={"format": "%Y-%m-%d"}, run_tokenization=False)]

    preprocessing_actions = [ConvertToCsv(SteamConverter()),
                             GroupAndFilter("items_filtered","product_id", GroupedFilter("count", lambda v: v >= min_item_feedback)),
                             GroupAndFilter("sessions_filtered","username", GroupedFilter("count", lambda v: v >= min_sequence_length)),
                             CreateSessionIndex(["username"]),
                             CreateRatioSplit(0.8, 0.1, 0.1,
                                              per_split_actions=
                                              [CreateSessionIndex(["username"]),
                                               CreateNextItemIndex(
                                                   [MetaInformation("item", column_name="product_id", type="str")],
                                                   RemainingSessionPositionExtractor(
                                                       min_sequence_length))],
                                              complete_split_actions=
                                              [CreateVocabulary(columns, prefixes=[prefix]),
                                               CreatePopularity(columns, prefixes=[prefix])]),
                             CreateLeaveOneOutSplit(MetaInformation("item", column_name="product_id", type="str"),
                                                    inner_actions=
                                                    [CreateNextItemIndex(
                                                        [MetaInformation("item", column_name="product_id", type="str")],
                                                        RemainingSessionPositionExtractor(
                                                            min_sequence_length)),
                                                        CreateVocabulary(columns),
                                                        CreatePopularity(columns)]),
                             CopyMainFile()
                             ]

    return DatasetPreprocessingConfig(prefix,
                                      "http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz",
                                      Path(output_directory),
                                      None,
                                      preprocessing_actions,
                                      context)


register_preprocessing_config_provider("steam",
                                       PreprocessingConfigProvider(get_steam_preprocessing_config,
                                                                   output_directory="./steam",
                                                                   min_item_feedback=4,
                                                                   min_sequence_length=4))


def get_yoochoose_preprocessing_config(output_directory: str,
                                       input_directory: str,
                                       min_item_feedback: int,
                                       min_sequence_length: int,
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

    preprocessing_actions = [ConvertToCsv(YooChooseConverter()),
                             GroupAndFilter("items_filtered", "ItemId", GroupedFilter("count", lambda v: v >= min_item_feedback)),
                             GroupAndFilter("sessions_filtered", "SessionId", GroupedFilter("count", lambda v: v >= min_sequence_length)),
                             CreateSessionIndex(["SessionId"]),
                             CreateRatioSplit(0.8, 0.1, 0.1,
                                              per_split_actions=
                                              [CreateSessionIndex(["SessionId"]),
                                               CreateNextItemIndex(
                                                   [MetaInformation("item", column_name="ItemId", type="str")],
                                                   RemainingSessionPositionExtractor(
                                                       min_sequence_length))],
                                              complete_split_actions=
                                              [CreateVocabulary(columns, prefixes=[prefix]),
                                               CreatePopularity(columns, prefixes=[prefix])]),
                             CreateLeaveOneOutSplit(MetaInformation("item", column_name="ItemId", type="str"),
                                                    inner_actions=
                                                    [CreateNextItemIndex(
                                                        [MetaInformation("item", column_name="ItemId", type="str")],
                                                        RemainingSessionPositionExtractor(
                                                            min_sequence_length)),
                                                        CreateVocabulary(columns),
                                                        CreatePopularity(columns)]),
                             CopyMainFile()]

    return DatasetPreprocessingConfig(prefix,
                                      None,
                                      Path(output_directory),
                                      None,
                                      preprocessing_actions,
                                      context)


register_preprocessing_config_provider("yoochoose",
                                       PreprocessingConfigProvider(get_yoochoose_preprocessing_config,
                                                                   output_directory="./yoochoose",
                                                                   min_item_feedback=4,
                                                                   min_sequence_length=4))


def get_example_preprocessing_config(output_directory: str,
                                     input_file_path: str,
                                     min_sequence_length: int) -> DatasetPreprocessingConfig:
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
               MetaInformation("attr_one", column_name="attr_one", type="str")]

    preprocessing_actions = [ConvertToCsv(ExampleConverter()),
                             CreateSessionIndex(["session_id"]),
                             CreateRatioSplit(0.8, 0.1, 0.1, per_split_actions=
                             [CreateSessionIndex(["session_id"]),
                              CreateNextItemIndex([MetaInformation("item", column_name="item_id", type="str")],
                                                  RemainingSessionPositionExtractor(min_sequence_length))],
                                              complete_split_actions=[
                                                  CreateVocabulary(columns, prefixes=[prefix]),
                                                  CreatePopularity(columns, prefixes=[prefix])]),
                             CreateLeaveOneOutSplit(MetaInformation("item", column_name="item_id", type="str"),
                                                    inner_actions=
                                                    [CreateNextItemIndex(
                                                        [MetaInformation("item", column_name="item_id", type="str")],
                                                        RemainingSessionPositionExtractor(
                                                            min_sequence_length)),
                                                        CreateVocabulary(columns),
                                                        CreatePopularity(columns)]),
                             CopyMainFile()
                             ]

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
                                                                   min_sequence_length=2))
