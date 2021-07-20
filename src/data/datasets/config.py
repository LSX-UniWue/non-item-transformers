from pathlib import Path

from asme.init.context import Context
from data.datamodule.config import DatasetPreprocessingConfig, register_preprocessing_config_provider, \
    PreprocessingConfigProvider
from data.datamodule.preprocessing import ConvertToCsv, TransformCsv, CreateSessionIndex, \
    GroupAndFilter, GroupedFilter, CreateVocabulary, DELIMITER_KEY, EXTRACTED_DIRECTORY_KEY, \
    OUTPUT_DIR_KEY, CreateRatioSplit, CreateNextItemIndex, CreateLeaveOneOutSplit, CreatePopularity, \
    RAW_INPUT_FILE_PATH_KEY, MAIN_FILE_KEY, UseExistingCsv, UseExistingSplit, SPLIT_BASE_DIRECTORY_PATH
from data.datamodule.column_info import ColumnInfo
from data.datamodule.converters import YooChooseConverter, Movielens1MConverter, ExampleConverter
from data.datamodule.extractors import RemainingSessionPositionExtractor
from data.datamodule.unpacker import Unzipper
from data.datamodule.preprocessing import PREFIXES_KEY
from data.datasets.sequence import MetaInformation


def get_ml_1m_preprocessing_config(output_directory: str,
                                   extraction_directory: str,
                                   min_item_feedback: int = 0,
                                   min_sequence_length: int = 2,
                                   ) -> DatasetPreprocessingConfig:
    prefix = "ml-1m"
    context = Context()
    context.set(PREFIXES_KEY, [prefix])
    context.set(DELIMITER_KEY, "\t")
    context.set(EXTRACTED_DIRECTORY_KEY, Path(extraction_directory))
    context.set(OUTPUT_DIR_KEY, Path(output_directory))

    special_tokens = ["<PAD>", "<MASK>", "<UNK>"]
    columns = [MetaInformation("rating", type="int"),
               MetaInformation("gender", type="str"),
               MetaInformation("age", type="int"),
               MetaInformation("occupation", type="str"),
               MetaInformation("zip", type="int"),
               MetaInformation("title", type="str"),
               MetaInformation("genres", type="str", configs={"delimiter": "|"})]

    preprocessing_actions = [ConvertToCsv(Movielens1MConverter()),
                             GroupAndFilter("movieId", GroupedFilter("count", lambda v: v >= min_item_feedback)),
                             GroupAndFilter("userId", GroupedFilter("count", lambda v: v >= min_sequence_length)),
                             CreateSessionIndex(["userId"]),
                             CreateRatioSplit(0.8, 0.1, 0.1,
                                              per_split_actions=
                                              [CreateSessionIndex(["userId"]),
                                               CreateNextItemIndex(
                                                   [MetaInformation("item", column_name="item_id", type="str")],
                                                   RemainingSessionPositionExtractor(
                                                       min_sequence_length))],
                                              complete_split_actions=
                                              [CreateVocabulary(columns, special_tokens=special_tokens,
                                                                prefixes=[prefix]),
                                               CreatePopularity(columns, prefixes=[prefix])]),
                             CreateLeaveOneOutSplit(MetaInformation("item", column_name="item_id", type="str"),
                                                    inner_actions=
                                                    [CreateNextItemIndex(
                                                        [MetaInformation("item", column_name="item_id", type="str")],
                                                        RemainingSessionPositionExtractor(
                                                            min_sequence_length)),
                                                        CreateVocabulary(columns, special_tokens=special_tokens),
                                                        CreatePopularity(columns)])]
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
                                                                   min_item_feedback=0,
                                                                   min_sequence_length=2))


def get_dota_shop_preprocessing_config(output_directory: str,
                                       raw_csv_file_path: str,
                                       split_directory: str,
                                       min_sequence_length: int) -> DatasetPreprocessingConfig:
    prefix = "dota"
    context = Context()
    context.set(PREFIXES_KEY, [prefix])
    context.set(DELIMITER_KEY, "\t")
    context.set(OUTPUT_DIR_KEY, Path(output_directory))

    full_csv_file_path = Path(raw_csv_file_path)
    context.set(RAW_INPUT_FILE_PATH_KEY, full_csv_file_path)

    # assume that the split directory is relative to the full csv file.
    split_directory_path = full_csv_file_path.parent / split_directory
    context.set(SPLIT_BASE_DIRECTORY_PATH, split_directory_path)

    special_tokens = ["<PAD>", "<MASK>", "<UNK>"]

    columns = [MetaInformation("hero_id", column_name="hero_id", type="str"),
               MetaInformation("item_id", column_name="item_id", type="str")]

    preprocessing_actions = [UseExistingCsv(),
                             CreateSessionIndex(["id", "hero_id"]),
                             CreateVocabulary(columns, special_tokens=special_tokens, prefixes=[prefix]),
                             UseExistingSplit(
                                 split_names=["train", "validation", "test"],
                                 per_split_actions=
                                 [
                                     CreateSessionIndex(["id", "hero_id"]),
                                     CreateNextItemIndex(
                                         [MetaInformation("item", column_name="item_id", type="str")],
                                         RemainingSessionPositionExtractor(min_sequence_length)
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
                                                                   min_sequence_length=2))


def get_example_preprocessing_config(output_directory: str,
                                     input_file_path: str,
                                     min_sequence_length: int) -> DatasetPreprocessingConfig:
    prefix = "example"
    context = Context()
    context.set(PREFIXES_KEY, [prefix])
    context.set(DELIMITER_KEY, "\t")
    context.set(OUTPUT_DIR_KEY, Path(output_directory))
    context.set(RAW_INPUT_FILE_PATH_KEY, input_file_path)

    special_tokens_mapping = {
        "pad_token": "<PAD>",
        "mask_token": "<MASK>",
        "unk_token": "<UNK>",
    }

    special_tokens = [token for _, token in special_tokens_mapping.items()]

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
                                                  CreateVocabulary(columns, special_tokens=special_tokens,
                                                                   prefixes=[prefix]),
                                                  CreatePopularity(columns, prefixes=[prefix],
                                                                   special_tokens=special_tokens_mapping)]),
                             CreateLeaveOneOutSplit(MetaInformation("item", column_name="item_id", type="str"),
                                                    inner_actions=
                                                    [CreateNextItemIndex(
                                                        [MetaInformation("item", column_name="item_id", type="str")],
                                                        RemainingSessionPositionExtractor(
                                                            min_sequence_length)),
                                                        CreateVocabulary(columns, special_tokens=special_tokens),
                                                        CreatePopularity(columns,
                                                                         special_tokens=special_tokens_mapping)])
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
