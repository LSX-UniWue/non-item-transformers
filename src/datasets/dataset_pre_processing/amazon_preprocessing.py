import pandas as pd
from pathlib import Path
from enum import Enum
from datasets.dataset_pre_processing.utils import download_dataset
import gzip
import json
import csv

AMAZON_DOWNLOAD_URL_MAP = {
    "games": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games.json.gz",
    "beauty": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty.json.gz"
}

AMAZON_ZIPPED_FILE_NAMES = {
"games": "reviews_Video_Games.json.gz",
"beauty": "reviews_Beauty.json.gz"
}

AMAZON_DELIMITER = "\t"
AMAZON_ITEM_ID = "product_id"
AMAZON_SESSION_ID = "reviewer_id"
AMAZON_REVIEW_TIMESTAMP_ID = "timestamp"
AMAZON_SESSION_COUNTS = "session_counts"
AMAZON_ITEM_COUNTS = "item_counts"


class FilterStrategy(str, Enum):
    """
    An enumeration that captures different strategies for how to apply filters to the datasets. Currently there are two
    different strategies used throughout the literature.
    1. joined: occurrences of values in several columns are first counted and then a filter is applied in one pass. Thus
               filtering of one column does not effect statistics for the other columns.
    2. pipelined: the filter for different columns are applied as a pipeline. Thus each filter has an impact on the
                  subsequents filters and thus the ordering is important.
    #TODO (AD): we should also add a 'core' strategy -> IMHO this is what everybody should be doing..
    """
    joined = "joined"
    pipelined = "sequential"


def convert_to_csv(input_file_path: Path,
                   output_file_path: Path,
                   delimiter: str = "\t"):
    """
    Extracts `reviewerID`, `asin` and `timestamp` from each line of the input file and stores the extracted data in csv
    format.

    :param input_file_path: an gzipped input file where every line is a json object with at least the above fields.
    :param output_file_path: a file where the csv data is written to.
    :param delimiter: the delimiter used when storing csv data.
    """
    with gzip.open(input_file_path) as file, output_file_path.open("w") as output_file:
        writer = csv.writer(output_file, delimiter=delimiter)
        writer.writerow([AMAZON_SESSION_ID, AMAZON_ITEM_ID, AMAZON_REVIEW_TIMESTAMP_ID])
        for line in file:
            parsed = json.loads(line)
            writer.writerow([parsed["reviewerID"], parsed["asin"], parsed["unixReviewTime"]])


def download_and_convert_amazon_dataset(category: str,
                                        output_dir: Path
                                        ) -> Path:
    """
    Downloads the Amazon dataset for the given category from http://jmcauley.ucsd.edu/data/amazon/links.html, extracts
    the relevant fields for each entry and stores the data in CSV format.

    :param category: a amazon category
    :param output_dir: a directory where the dataset will be stored

    :return the path to the csv file
    """
    csv_file_path = output_dir / f"{category}.csv"
    download_dataset(AMAZON_DOWNLOAD_URL_MAP[category], output_dir)
    convert_to_csv(input_file_path=output_dir.joinpath(AMAZON_ZIPPED_FILE_NAMES[category]),
                   output_file_path=csv_file_path,
                   delimiter=AMAZON_DELIMITER)
    return csv_file_path


def filter_category_occurrences(df: pd.DataFrame,
                                filter_category: str,
                                min_occurrences: int = 5
                                ) -> pd.DataFrame:
    # rare products
    df_counts = df.groupby([filter_category]).count()
    df_counts = df_counts[df_counts["timestamp"] >= min_occurrences]
    df_counts = df_counts.reset_index()
    product_ids = df_counts[filter_category].unique().tolist()
    return df[df[filter_category].isin(product_ids)]


def filter_pipelined(df: pd.DataFrame,
                     min_occurrences: int = 5
                     ) -> pd.DataFrame:
    """
    Filter min session length and min item occurrence as a pipeline with order: item occurrence, session length

    :param df: a data frame
    :param min_occurrences: the minimum number of occurrences.

    :return: the filtered data frame.
    """
    # remove all items with less then min_occurrences reviews
    df = filter_category_occurrences(df, AMAZON_ITEM_ID, min_occurrences=min_occurrences)
    # remove all sessions with less then min_occurrences reviews
    df = filter_category_occurrences(df, AMAZON_SESSION_ID, min_occurrences=min_occurrences)

    return df


def filter_joined(df: pd.DataFrame,
                  min_occurrences: int = 5
                  ) -> pd.DataFrame:
    """
    Filter both min session length and min item occurrences in one pass.

    :param df: a data frame
    :param min_occurrences: minimum item occurrence.

    :return: filtered data frame.
    """
    product_counts = df.groupby([AMAZON_ITEM_ID]).count().rename(columns={AMAZON_SESSION_ID: AMAZON_ITEM_COUNTS}).drop(columns="timestamp")
    reviewer_counts = df.groupby([AMAZON_SESSION_ID]).count().rename(columns={AMAZON_ITEM_ID: AMAZON_SESSION_COUNTS}).drop(columns="timestamp")

    m1 = pd.merge(df, reviewer_counts, on=[AMAZON_SESSION_ID])
    m2 = pd.merge(m1, product_counts, on=[AMAZON_ITEM_ID])
    df_filtered = m2.loc[(m2[AMAZON_SESSION_COUNTS] >= min_occurrences) & (m2[AMAZON_ITEM_COUNTS] >= min_occurrences)]

    return df_filtered.drop(columns=[AMAZON_SESSION_COUNTS, AMAZON_ITEM_COUNTS]).sort_values(by=[AMAZON_SESSION_ID])


def preprocess_amazon_dataset_for_indexing(input_file_path: Path,
                                           filter_strategy: FilterStrategy,
                                           output_file_prefix: str = "preprocessed-", min_occurrences: int = 5
                                           ) -> Path:

    # read only the data we need into memory
    df = pd.read_csv(filepath_or_buffer=input_file_path,
                     delimiter=AMAZON_DELIMITER,
                     usecols=[AMAZON_SESSION_ID, AMAZON_ITEM_ID, AMAZON_REVIEW_TIMESTAMP_ID])

    # make sure that sessions are grouped together within the dataframe and ordered by timestamp
    df = df.sort_values(by=[AMAZON_SESSION_ID, AMAZON_REVIEW_TIMESTAMP_ID])

    if filter_strategy == FilterStrategy.joined:
        df = filter_joined(df, min_occurrences)
    elif filter_strategy == FilterStrategy.pipelined:
        df = filter_pipelined(df, min_occurrences)

    output_file_path = input_file_path.parent / f"{output_file_prefix}{input_file_path.name}"
    df.to_csv(output_file_path, sep=AMAZON_DELIMITER, index=False)

    return output_file_path
