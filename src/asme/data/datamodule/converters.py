import os
import shutil
from abc import abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from asme.datasets.dataset_pre_processing.utils import read_csv
import json
import csv
import gzip


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
    YOOCHOOSE_SESSION_ID_KEY = "SessionId"

    def __init__(self, delimiter="\t"):
        self.delimiter = delimiter

    def apply(self, input_dir: Path, output_file: Path):

        data = pd.read_csv(input_dir.joinpath('yoochoose-clicks.dat'),
                           sep=',',
                           header=None,
                           usecols=[0, 1, 2],
                           dtype={0: np.int32, 1: str, 2: np.int64},
                           names=['SessionId', 'TimeStr', 'ItemId'])

        data['Time'] = data.TimeStr.apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())
        data = data.drop("TimeStr", axis=1)

        if not os.path.exists(output_file):
            output_file.parent.mkdir(parents=True, exist_ok=True)
        data = data.sort_values(self.YOOCHOOSE_SESSION_ID_KEY)
        data.to_csv(path_or_buf=output_file, sep=self.delimiter, index=False)


class Movielens20MConverter(CsvConverter):
    RATING_USER_COLUMN_NAME = 'userId'
    RATING_MOVIE_COLUMN_NAME = 'movieId'
    RATING_TIMESTAMP_COLUMN_NAME = 'timestamp'

    def __init__(self, delimiter="\t"):
        self.delimiter = delimiter

    def apply(self, input_dir: Path, output_file: Path):
        file_type = ".csv"
        header = 0
        sep = ","
        name = "ml-20m"
        location = input_dir / name
        ratings_df = read_csv(location, "ratings", file_type, sep, header)

        movies_df = read_csv(location, "movies", file_type, sep, header)

        links_df = read_csv(location, "links", file_type, sep, header)
        ratings_df = pd.merge(ratings_df, links_df)

        merged_df = pd.merge(ratings_df, movies_df).sort_values(
            by=[Movielens20MConverter.RATING_USER_COLUMN_NAME, Movielens20MConverter.RATING_TIMESTAMP_COLUMN_NAME])

        # Remove unnecessary columns, we keep movieId here so that we can filter later.
        merged_df = merged_df.drop('imdbId', axis=1).drop('tmdbId', axis=1)

        os.makedirs(output_file.parent, exist_ok=True)

        merged_df.to_csv(output_file, sep=self.delimiter, index=False)


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
        encoding = "latin-1"
        ratings_df = read_csv(location, "ratings", file_type, sep, header, encoding=encoding)

        ratings_df.columns = [Movielens1MConverter.RATING_USER_COLUMN_NAME,
                              Movielens1MConverter.RATING_MOVIE_COLUMN_NAME, 'rating',
                              Movielens1MConverter.RATING_TIMESTAMP_COLUMN_NAME]

        movies_df = read_csv(location, "movies", file_type, sep, header, encoding=encoding)

        movies_df.columns = ['movieId', 'title', 'genres']
        users_df = read_csv(location, "users", file_type, sep, header, encoding=encoding)
        users_df.columns = [Movielens1MConverter.RATING_USER_COLUMN_NAME, 'gender', 'age', 'occupation', 'zip']
        ratings_df = pd.merge(ratings_df, users_df)

        merged_df = pd.merge(ratings_df, movies_df).sort_values(
            by=[Movielens1MConverter.RATING_USER_COLUMN_NAME, Movielens1MConverter.RATING_TIMESTAMP_COLUMN_NAME])

        os.makedirs(output_file.parent, exist_ok=True)

        merged_df.to_csv(output_file, sep=self.delimiter, index=False)


class AmazonConverter(CsvConverter):
    AMAZON_SESSION_ID = "reviewer_id"
    AMAZON_ITEM_ID = "product_id"
    AMAZON_REVIEW_TIMESTAMP_ID = "timestamp"

    def __init__(self, delimiter="\t"):
        self.delimiter = delimiter

    def apply(self, input_dir: Path, output_file: Path):
        os.makedirs(output_file.parent, exist_ok=True)
        with gzip.open(input_dir) as file, output_file.open("w") as output_file:
            rows = []
            for line in file:
                parsed = json.loads(line)
                rows.append([parsed["reviewerID"], parsed["asin"], parsed["unixReviewTime"]])

            df = pd.DataFrame(rows, columns=[self.AMAZON_SESSION_ID,
                                             self.AMAZON_ITEM_ID,
                                             self.AMAZON_REVIEW_TIMESTAMP_ID])
            df = df.sort_values(by=[self.AMAZON_SESSION_ID, self.AMAZON_REVIEW_TIMESTAMP_ID])
            df.to_csv(output_file, sep=self.delimiter, index=False)


class SteamConverter(CsvConverter):
    STEAM_SESSION_ID = "username"
    STEAM_ITEM_ID = "product_id"
    STEAM_TIMESTAMP = "date"

    def __init__(self, delimiter="\t"):
        self.delimiter = delimiter

    def apply(self, input_dir: Path, output_file: Path):

        if not output_file.parent.exists():
            os.makedirs(output_file.parent, exist_ok=True)

        with gzip.open(input_dir, mode="rt") as input_file:
            rows = []
            for record in input_file:
                parsed_record = eval(record)
                username = parsed_record[self.STEAM_SESSION_ID]
                product_id = int(parsed_record[self.STEAM_ITEM_ID])
                timestamp = parsed_record[self.STEAM_TIMESTAMP]

                row = [username, product_id, timestamp]
                rows.append(row)

        df = pd.DataFrame(rows, columns=[self.STEAM_SESSION_ID,
                                         self.STEAM_ITEM_ID,
                                         self.STEAM_TIMESTAMP])
        df = df.sort_values(by=[self.STEAM_SESSION_ID, self.STEAM_TIMESTAMP])
        df.to_csv(output_file, sep=self.delimiter, index=False)


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
