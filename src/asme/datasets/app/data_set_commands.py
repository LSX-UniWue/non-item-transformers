import os
import typer
from pathlib import Path
from typing import List

from asme.datasets.data_structures.dataset_metadata import DatasetMetadata
from asme.datasets.data_structures.split_strategy import SplitStrategy
from asme.datasets.dataset_index_splits import split_strategies_factory
from asme.datasets.dataset_pre_processing.generic import generic_process_dataset

from asme.datasets.dataset_pre_processing.movielens_preprocessing import download_and_unzip_movielens_data, \
    preprocess_movielens_data, MOVIELENS_DELIMITER, MOVIELENS_ITEM_HEADER_NAME, MOVIELENS_SESSION_KEY, \
    RATING_USER_COLUMN_NAME
from asme.datasets.dataset_pre_processing.utils import download_dataset
from asme.datasets.dataset_pre_processing.yoochoose_preprocessing import pre_process_yoochoose_dataset, \
    YOOCHOOSE_CLICKS_FILE_NAME, YOOCHOOSE_SESSION_ID_KEY, YOOCHOOSE_ITEM_ID_KEY, YOOCHOOSE_BUYS_FILE_NAME, \
    YOOCHOOSE_DELIMITER
from asme.datasets.dataset_pre_processing.amazon_preprocessing import download_and_convert_amazon_dataset, AMAZON_ITEM_ID, \
    AMAZON_SESSION_ID, AMAZON_DELIMITER, preprocess_amazon_dataset_for_indexing, filter_category_occurrences, \
    FilterStrategy

app = typer.Typer()


DEFAULT_SPECIAL_TOKENS = ["<PAD>", "<MASK>", "<UNK>"]


@app.command()
def movielens(dataset: str = typer.Argument(..., help="ml-1m or ml-20m", show_choices=True),
              output_dir: Path = typer.Option("./dataset/", help='directory to save data'),
              min_user_feedback: int = typer.Option(0, help='the minimum number of feedback a user must have'),
              min_item_feedback: int = typer.Option(0, help='the minimum number of feedback an item must have received')
              ) -> None:
    dataset_dir, extract_dir = download_and_unzip_movielens_data(dataset,
                                                                 output_dir)
    main_file = preprocess_movielens_data(extract_dir,
                                          dataset_dir,
                                          dataset,
                                          min_item_feedback=min_item_feedback,
                                          min_user_feedback=min_user_feedback)

    stats_columns = ["title", "genres", RATING_USER_COLUMN_NAME]
    if dataset == "ml-1m":
        stats_columns += ["gender", "age", "occupation", "zip"]

    dataset_metadata = DatasetMetadata(
        data_file_path=main_file,
        session_key=[MOVIELENS_SESSION_KEY],
        item_header_name=MOVIELENS_ITEM_HEADER_NAME,
        delimiter=MOVIELENS_DELIMITER,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
        stats_columns=stats_columns
    )

    split_strategies = _build_split_strategies(['loo', 'ratio'], 0.95, 0.05, 0.05)

    generic_process_dataset(dataset_metadata=dataset_metadata, split_strategies=split_strategies)


@app.command()
def steam(dataset_path: Path = typer.Argument("./dataset/steam", help="directory where the dataset will be stored."),
          min_seq_length: int = typer.Option(5, help="The minimum length of a session to be considered for the dataset.")):

    # download data if necessary
    reviews_url = "http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz"
    game_info_url = "http://cseweb.ucsd.edu/~wckang/steam_games.json.gz"

    reviews_file_path = dataset_path / "steam_reviews.json.gz"
    game_info_file_path = dataset_path / "steam_games.json.gz"

    if not reviews_file_path.exists():
        print("Downloading reviews file")
        download_dataset(reviews_url, dataset_path)

    if not game_info_file_path.exists():
        print("Downloading game info file")
        download_dataset(game_info_url, dataset_path)

    # create csv dataset
    def create_csv_dataset(input_file_path: Path, output_file_path: Path):
        """
        Parses the file with reviews and extracts `username`, `product_id` and `date` of each entry into a csv file.

        :param input_file_path: the file with steam reviews (not really json).
        :param output_file_path: the csv file where the dataset is written.
        """
        import gzip
        import csv
        with gzip.open(input_file_path, mode="rt") as input_file, output_file_path.open("w") as output_file:
            writer = csv.writer(output_file, delimiter="\t")
            writer.writerow(["username", "product_id", "timestamp"])
            for record in input_file:
                parsed_record = eval(record)
                username = parsed_record["username"]
                product_id = int(parsed_record["product_id"])
                timestamp = parsed_record["date"]

                row = [username, product_id, timestamp]
                writer.writerow(row)

    output_file_path = dataset_path / "steam.csv"
    if not output_file_path.exists():
        print("Extracting review sessions")
        create_csv_dataset(reviews_file_path, output_file_path)

    # preprocess dataset
    def preprocess_dataset(input_file_path: Path, output_file_path: Path, min_occurrences: int):
        import pandas as pd
        raw_df = pd.read_csv(input_file_path, delimiter="\t", usecols=["username", "product_id", "timestamp"])
        df = raw_df.sort_values(by=["username", "timestamp"])
        df = filter_category_occurrences(df, "product_id", min_occurrences = min_occurrences)
        df = filter_category_occurrences(df, "username", min_occurrences = min_occurrences)

        df.to_csv(output_file_path, sep="\t", index=False)

    preprocessed_output_file_path = dataset_path / f"preprocessed-{output_file_path.name}"
    if not preprocessed_output_file_path.exists():
        preprocess_dataset(output_file_path, preprocessed_output_file_path, min_seq_length)

    # split and generate indices
    dataset_metadata = DatasetMetadata(
        data_file_path=preprocessed_output_file_path,
        session_key=["username"],
        item_header_name="product_id",
        delimiter="\t",
        special_tokens=DEFAULT_SPECIAL_TOKENS,
        stats_columns=["product_id"]
    )

    split_strategies = _build_split_strategies(['loo', 'ratio'], 0.95, 0.05, 0.05)

    generic_process_dataset(dataset_metadata=dataset_metadata, split_strategies=split_strategies)


@app.command()
def yoochoose(input_dir: Path = typer.Argument("./dataset/yoochoose-data",
                                               help='directory path to the raw yoochoose data set'),
              output_dir_path: Path = typer.Argument("./dataset/yoochoose/",
                                                     help='Output directory for indices, splits, and vocabulary.'),
              category: str = typer.Option(..., help="buys or clicks"),
              min_seq_length: int = typer.Option(5, help='The minimum length of a session for the next item split')
              ) -> None:
    """
    Handles pre-processing, splitting, storing and indexing of yoochoose data set.
    As a prerequisite the yoochoose data set has to have been downloaded and stored at the input_dir.

    :param input_dir: Directory under which the yoochoose data set files are stored.
    :param output_dir_path: Directory under which the processed data is stored after successful execution.
    :param category: buys or clicks depending on which data should be processed
    :param min_seq_length: Minimum length of a session in order to be included in the next item split
    :return:
    """
    if category == "buys":
        file_name = YOOCHOOSE_BUYS_FILE_NAME
    elif category == "clicks":
        file_name = YOOCHOOSE_CLICKS_FILE_NAME
    else:
        raise Exception("Please specify a correct --category=[clicks or buys]")
    output_dir_path = output_dir_path / category
    # Check if input dir contains the correct data path and that the yoochoose dataset is downloaded
    if not os.path.isfile(input_dir / (file_name + '.dat')):
        print(input_dir / (file_name + '.dat'),
              "does not exist. Please download the yoochoose data set and move it into", input_dir, ".")
        print("See --help for more information.")
    else:
        # Pre-process yoochoose data
        print("Perform pre-processing...")
        preprocessed_data_filepath = pre_process_yoochoose_dataset(input_dir, output_dir_path, file_name=file_name)

        print("Creating necessary files for training and evaluation...")
        dataset_metadata = DatasetMetadata(
            data_file_path=preprocessed_data_filepath,
            session_key=[YOOCHOOSE_SESSION_ID_KEY],
            item_header_name=YOOCHOOSE_ITEM_ID_KEY,
            delimiter=YOOCHOOSE_DELIMITER,
        )

        split_strategies = _build_split_strategies(['loo', 'ratio'], 0.95, 0.05, 0.05)
        generic_process_dataset(dataset_metadata=dataset_metadata, split_strategies=split_strategies)


def _build_split_strategies(splits_to_generate: List[str],
                            train_ratio: float,
                            validation_ratio: float,
                            test_ratio: float
                            ) -> List[SplitStrategy]:
    split_strategy = []
    for split_id in splits_to_generate:
        if 'loo' == split_id:
            continue  # TODO: add the loo strategy
        if 'ratio' == split_id:
            ratio_split = split_strategies_factory.get_ratio_strategy(train_ratio=train_ratio,
                                                                      validation_ratio=validation_ratio,
                                                                      test_ratio=test_ratio,
                                                                      seed=123456)
            split_strategy.append(ratio_split)
    return split_strategy


@app.command()
def amazon(output_dir_path: Path = typer.Argument("./dataset/amazon/",
                                                  help='Output directory for indices, splits, and vocabulary.'),
           category: str = typer.Option(..., help="beauty or games"),
           min_occurrences: int = typer.Option(5, help='The minimum number of occurrences used to filter'
                                                       'infrequently used items and short sessions.'),
           filter_strategy: FilterStrategy = typer.Option(FilterStrategy.pipelined, help="the strategy used to apply"
                                                                                         "filters to the dataset.")
           ) -> None:
    """
    Handles pre-processing, splitting, storing and indexing of amazon data sets.

    :param output_dir_path: Directory under which the processed data is stored after successful execution.
    :param category: Amazon reviews category
    :param min_occurrences: The minimum number of occurrences used to filter infrequently used items and short sessions.
    :param filter_strategy: the strategy used to apply filters to the dataset.
    :return:
    """
    output_dir_path = output_dir_path / category
    # Pre-process yoochoose data
    print("Download dataset...")
    raw_data_file_path = download_and_convert_amazon_dataset(category=category, output_dir=output_dir_path)

    print("Pre-process data...")
    processed_data_file_path = preprocess_amazon_dataset_for_indexing(
        input_file_path=raw_data_file_path,
        filter_strategy=filter_strategy,
        min_occurrences=min_occurrences
    )

    print("Creating necessary files for training and evaluation...")

    dataset_metadata = DatasetMetadata(
        data_file_path=processed_data_file_path,
        session_key=[AMAZON_SESSION_ID],
        item_header_name=AMAZON_ITEM_ID,
        delimiter=AMAZON_DELIMITER,
        special_tokens=DEFAULT_SPECIAL_TOKENS
    )

    split_strategies = _build_split_strategies(['loo', 'ratio'], 0.95, 0.05, 0.05)

    generic_process_dataset(dataset_metadata=dataset_metadata, split_strategies=split_strategies)


@app.command()
def generic_csv_file(csv_file: Path = typer.Argument(..., help='path to the csv file'),
                     session_key: List[str] = typer.Argument(..., help='the session key'),
                     item_header_name: str = typer.Argument(..., help='item header name'),
                     delimiter: str = typer.Option('\t', help='the delimiter to use for csv'),
                     splits_to_generate: List[str] = typer.Option(['loo', 'ratio'], help=''),
                     train_ratio: float = typer.Option(0.8, help='the train ratio'),
                     validation_ratio: float = typer.Option(0.1, help='the validation ratio'),
                     test_ratio: float = typer.Option(0.1, help='the test ratio'),

            ) -> None:
    """
    Generates the configured splits for the specified csv file

    :param csv_file:
    :param session_key:
    :param item_header_name:
    :param delimiter:
    :param splits_to_generate:
    :param train_ratio:
    :param validation_ratio:
    :param test_ratio:
    :return:
    """
    dataset_metadata = DatasetMetadata(
        data_file_path=csv_file,
        session_key=session_key,
        item_header_name=item_header_name,
        delimiter=delimiter,
        special_tokens=DEFAULT_SPECIAL_TOKENS
    )

    split_strategies = _build_split_strategies(splits_to_generate, train_ratio, validation_ratio, test_ratio)

    generic_process_dataset(dataset_metadata=dataset_metadata, split_strategies=split_strategies)
