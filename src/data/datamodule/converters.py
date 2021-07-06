import os
import shutil
from abc import abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.dataset_pre_processing.utils import read_csv


class CsvConverter:
    """
    Base class for all dataset converters. Subtypes of this class should be able to convert a specific dataset into a
    single CSV file.
    """
    @abstractmethod
    def apply(self, input_dir: Path, output_file: Path):
        """
        Converts the dataset into a single CSV file and saves it at output_file.

        :param input_dir: The path to the file/directory of the dataset.
        :param output_file: The path to the resulting CSV file.
        """
        pass

    def __call__(self, input_dir: Path, output_file: Path):
        return self.apply(input_dir, output_file)


class YooChooseConverter(CsvConverter):
    def apply(self, input_dir: Path, output_file: Path):
        YOOCHOOSE_SESSION_ID_KEY = "SessionId"
        YOOCHOOSE_ITEM_ID_KEY = "ItemId"

        data = pd.read_csv(input_dir.joinpath('clicks.dat'),
                           sep=',',
                           header=None,
                           usecols=[0, 1, 2],
                           dtype={0: np.int32, 1: str, 2: np.int64},
                           names=['SessionId', 'TimeStr', 'ItemId'])

        data['Time'] = data.TimeStr.apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())
        session_lengths = data.groupby(YOOCHOOSE_SESSION_ID_KEY).size()
        data = data[np.in1d(data.SessionId, session_lengths[session_lengths > 1].index)]
        item_supports = data.groupby(YOOCHOOSE_ITEM_ID_KEY).size()
        data = data[np.in1d(data.ItemId, item_supports[item_supports >= 5].index)]
        session_lengths = data.groupby(YOOCHOOSE_SESSION_ID_KEY).size()
        data = data[np.in1d(data.SessionId, session_lengths[session_lengths >= 2].index)]

        if not os.path.exists(output_file):
            output_file.parent.mkdir(parents=True, exist_ok=True)
        data = data.sort_values(YOOCHOOSE_SESSION_ID_KEY)
        data.to_csv(path_or_buf=output_file)


class Movielens1MConverter(CsvConverter):
    RATING_USER_COLUMN_NAME = 'userId'
    RATING_MOVIE_COLUMN_NAME = 'movieId'
    RATING_TIMESTAMP_COLUMN_NAME = 'timestamp'

    def __init__(self, delimiter="\t"):
        self.delimiter = delimiter

    def apply(self, input_dir: Path, output_file: Path):
        file_type = ".dat"
        header = None
        sep = "::"
        name = "ml-1m"
        location = input_dir / name
        ratings_df = read_csv(location, "ratings", file_type, sep, header)

        ratings_df.columns = [Movielens1MConverter.RATING_USER_COLUMN_NAME,
                              Movielens1MConverter.RATING_MOVIE_COLUMN_NAME, 'rating',
                              Movielens1MConverter.RATING_TIMESTAMP_COLUMN_NAME]

        movies_df = read_csv(location, "movies", file_type, sep, header)

        movies_df.columns = ['movieId', 'title', 'genres']
        users_df = read_csv(location, "users", file_type, sep, header)
        users_df.columns = [Movielens1MConverter.RATING_USER_COLUMN_NAME, 'gender', 'age', 'occupation', 'zip']
        ratings_df = pd.merge(ratings_df, users_df)

        merged_df = pd.merge(ratings_df, movies_df).sort_values(
            by=[Movielens1MConverter.RATING_USER_COLUMN_NAME, Movielens1MConverter.RATING_TIMESTAMP_COLUMN_NAME])

        os.makedirs(output_file.parent, exist_ok=True)

        merged_df.to_csv(output_file, sep=self.delimiter, index=False)


class DotaShopConverter(CsvConverter):

    def __init__(self):
        pass

    def apply(self, input_dir: Path, output_file: Path):
        # We assume `input_dir` to be the path to the raw csv file.
        shutil.copy(input_dir, output_file)


class ExampleConverter(CsvConverter):

    def __init__(self):
        pass

    def apply(self, input_dir: Path, output_file: Path):
        # We assume `input_dir` to be the path to the raw csv file.
        shutil.copy(input_dir, output_file)

