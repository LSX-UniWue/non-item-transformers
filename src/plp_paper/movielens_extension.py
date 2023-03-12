from pathlib import Path

from asme.core.init.context import Context
from asme.core.init.templating.datasources.datasources import DatasetSplit
from asme.data import CURRENT_SPLIT_PATH_CONTEXT_KEY
from asme.data.datamodule.config import DatasetPreprocessingConfig
from asme.data.datamodule.preprocessing.action import PREFIXES_KEY, DELIMITER_KEY, INPUT_DIR_KEY, OUTPUT_DIR_KEY
from asme.data.datamodule.preprocessing.indexing import CreateSessionIndex
from asme.data.datamodule.preprocessing.popularity import CreatePopularity
from asme.data.datamodule.preprocessing.split import UseExistingSplit
from asme.data.datamodule.preprocessing.vocabulary import CreateVocabulary
from asme.data.datasets.sequence import MetaInformation


def get_movielens_extended_preprocessing_config(
        # General parameters
        output_directory: str,
        extraction_directory: str,
        prefix="ml-1m-extended"
) -> DatasetPreprocessingConfig:
    context = Context()
    context.set(PREFIXES_KEY, [prefix])
    context.set(DELIMITER_KEY, "\t")
    context.set(INPUT_DIR_KEY, Path(extraction_directory))
    context.set(OUTPUT_DIR_KEY, Path(output_directory))

    context = Context()
    context.set(PREFIXES_KEY, [prefix])
    context.set(DELIMITER_KEY, "\t")
    context.set(OUTPUT_DIR_KEY, Path(output_directory))
    context.set(CURRENT_SPLIT_PATH_CONTEXT_KEY, Path(extraction_directory))


    columns = [MetaInformation("rating", type="int", run_tokenization=False),
               MetaInformation("gender", type="str"),
               MetaInformation("age", type="str"),
               MetaInformation("occupation", type="str"),
               MetaInformation("title", type="str"),
               MetaInformation("zip", type="str"),
               MetaInformation("title_genres", type="str"),
               MetaInformation("title_uid", type="str"),
               MetaInformation("userId", type="str"),
               MetaInformation("user_all", type="list", configs={"delimiter": "|", "element_type": "str"}),
               MetaInformation("year", type="str", run_tokenization=False),
               MetaInformation("genres", type="str", configs={"delimiter": "|"})]


    item_column = MetaInformation("item", column_name="title", type="str")
    min_item_feedback_column = "movieId"
    min_sequence_length_column = "userId"
    session_key = ["userId"]

    per_split_actions = [
        CreateSessionIndex(["userId"]),
        CreateVocabulary(columns, prefixes=[prefix]),
        CreatePopularity(columns, prefixes=[prefix])]

    preprocessing_actions = [UseExistingSplit(
                                 split_names=["train", "test", "validation"],
                                 split_type=DatasetSplit.RATIO_SPLIT,
                                 per_split_actions=per_split_actions)
                             ]

    return DatasetPreprocessingConfig(prefix,
                                      None,
                                      Path(output_directory),
                                      None,
                                      preprocessing_actions,
                                      context)



