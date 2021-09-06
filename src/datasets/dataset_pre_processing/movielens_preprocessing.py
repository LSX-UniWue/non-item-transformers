import os
import pandas as pd
from pathlib import Path

from datasets.dataset_pre_processing.utils import read_csv, download_dataset, unzip_file

RATING_USER_COLUMN_NAME = 'userId'
RATING_MOVIE_COLUMN_NAME = 'movieId'
RATING_TIMESTAMP_COLUMN_NAME = 'timestamp'

MOVIELENS_SESSION_KEY = RATING_USER_COLUMN_NAME
MOVIELENS_ITEM_HEADER_NAME = "title"
MOVIELENS_DELIMITER = "\t"

DOWNLOAD_URL_MAP = {
    'ml-1m': 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
    'ml-20m': 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'
}


def download_and_unzip_movielens_data(dataset: str,
                                      output_dir: Path) -> (Path, Path):
    url = DOWNLOAD_URL_MAP[dataset]
    dataset_dir = output_dir / f'{dataset}'
    download_dir = output_dir

    downloaded_file = download_dataset(url, download_dir)

    extract_dir = download_dir / 'raw'
    if not os.path.exists(extract_dir):
        extract_dir.mkdir(parents=True, exist_ok=True)

    unzip_file(downloaded_file, extract_dir, delete=False)
    return dataset_dir, extract_dir


def filter_ratings(ratings_df: pd.DataFrame,
                   min_user_feedback: int = 0,
                   min_item_feedback: int = 0
                   ):
    def _filter_dataframe(column: str, min_count: int, dataframe: pd.DataFrame) -> pd.DataFrame:
        sizes = ratings_df.groupby(column).size()
        good_entities = sizes.index[sizes >= min_count]

        return dataframe[ratings_df[column].isin(good_entities)]

    # (AD) we adopt the order used in bert4rec preprocessing
    if min_item_feedback > 1:
        ratings_df = _filter_dataframe(RATING_MOVIE_COLUMN_NAME, min_item_feedback, ratings_df)

    if min_user_feedback > 1:
        ratings_df = _filter_dataframe(RATING_USER_COLUMN_NAME, min_user_feedback, ratings_df)

    return ratings_df


def preprocess_movielens_data(dataset_dir: Path,
                              output_dir: Path,
                              name: str,
                              delimiter: str = '\t',
                              min_user_feedback: int = 0,
                              min_item_feedback: int = 0
                              ) -> Path:
    """
    Convert raw movielens data to csv files and create vocabularies
    :param min_item_feedback:
    :param delimiter:
    :param output_dir:
    :param dataset_dir:
    :param name:
    :param min_user_feedback:
    :return: the path to the main file
    """
    print("Convert to csv...")

    if name == "ml-1m":
        file_type = ".dat"
        header = None
        sep = "::"
    else:
        file_type = ".csv"
        header = 0
        sep = ","

    dataset_dir = dataset_dir / name
    # read and merge data
    print("Dataset dir", dataset_dir)
    ratings_df = read_csv(dataset_dir, "ratings", file_type, sep, header)

    movielens_1m = name == 'ml-1m'
    if movielens_1m:
        ratings_df.columns = [RATING_USER_COLUMN_NAME,
                              RATING_MOVIE_COLUMN_NAME, 'rating',
                              RATING_TIMESTAMP_COLUMN_NAME]

    ratings_df = filter_ratings(ratings_df,
                                min_user_feedback=min_user_feedback,
                                min_item_feedback=min_item_feedback)

    movies_df = read_csv(dataset_dir, "movies", file_type, sep, header)

    # only the ml-1m dataset has got a user info file …

    if movielens_1m:
        movies_df.columns = ['movieId', 'title', 'genres']
        users_df = read_csv(dataset_dir, "users", file_type, sep, header)
        users_df.columns = [RATING_USER_COLUMN_NAME, 'gender', 'age', 'occupation', 'zip']
        ratings_df = pd.merge(ratings_df, users_df)
    elif name == "ml-20":
        links_df = read_csv(dataset_dir, "links", file_type, sep, header)
        ratings_df = pd.merge(ratings_df, links_df)

    merged_df = pd.merge(ratings_df, movies_df).sort_values(by=[RATING_USER_COLUMN_NAME, RATING_TIMESTAMP_COLUMN_NAME])
    # remove unused movie id column
    del merged_df[RATING_MOVIE_COLUMN_NAME]

    os.makedirs(output_dir, exist_ok=True)

    main_file = output_dir / f"{name}.csv"
    merged_df.to_csv(main_file, sep=delimiter, index=False)

    return main_file
