import ast
import os
import shutil
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Any, Dict
from tqdm import tqdm
import time

import numpy as np
import pandas as pd

from asme.data.datamodule.util import read_csv
import json
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
        movies_df["year"] = movies_df["title"].str.rsplit(r"(", 1).apply(lambda x: x[1].rsplit(r")")[0]).astype(int)
        users_df = read_csv(location, "users", file_type, sep, header, encoding=encoding)
        users_df.columns = [Movielens1MConverter.RATING_USER_COLUMN_NAME, 'gender', 'age', 'occupation', 'zip']
        ratings_df = pd.merge(ratings_df, users_df)

        merged_df = pd.merge(ratings_df, movies_df).sort_values(
            by=[Movielens1MConverter.RATING_USER_COLUMN_NAME, Movielens1MConverter.RATING_TIMESTAMP_COLUMN_NAME])

        os.makedirs(output_file.parent, exist_ok=True)
        merged_df["user_all"] = merged_df["gender"].astype(str) + "|" + merged_df["age"].astype(str) + "age|" + \
                                merged_df["occupation"].astype(str) + "occupation"
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


class Track:
    def __init__(self, name: str, album: str, artist: str, genre: str = None):
        self.name = name
        self.album = album
        self.artist = artist
        self.genre = genre

    def __getitem__(self, key):
        return getattr(self, key)


class SpotifyConverter(CsvConverter):
    RAW_TRACKS_KEY = "tracks"
    RAW_TIMESTAMP_KEY = "modified_at"
    RAW_PLAYLIST_ID_KEY = "pid"
    _SPOTIFY_TIME_COLUMN = "playlist_timestamp"
    SPOTIFY_SESSION_ID = "playlist_id"
    SPOTIFY_ITEM_ID = "track_name"
    SPOTIFY_ALBUM_NAME_KEY = "album_name"
    SPOTIFY_ARTIST_NAME_KEY = "artist_name"

    # SPOTIFY_DATETIME_PARSER = DateTimeParser(time_column_name=_SPOTIFY_TIME_COLUMN,
    #                                          date_time_parse_function=lambda x: datetime.fromtimestamp(int(x)))

    def __init__(self, delimiter="\t"):
        self.delimiter = delimiter

    def _process_playlist(self, playlist: Dict) -> List[Track]:
        tracks_list: List[Track] = []
        for track in playlist[self.RAW_TRACKS_KEY]:
            track_name: str = track[self.SPOTIFY_ITEM_ID]
            album_name: str = track[self.SPOTIFY_ALBUM_NAME_KEY]
            artist_name: str = track[self.SPOTIFY_ARTIST_NAME_KEY]
            tracks_list += [Track(name=track_name, album=album_name, artist=artist_name)]
        return tracks_list

    def apply(self, input_dir: Path, output_file: Path):
        dataset: List[List[Any]] = []
        filenames = os.listdir(input_dir)
        for filename in tqdm(sorted(filenames), desc=f"Process playlists in file"):
            if filename.startswith("mpd.slice.") and filename.endswith(".json"):
                file_path: Path = input_dir.joinpath(filename)
                f = open(file_path)
                js = f.read()
                f.close()
                mpd_slice = json.loads(js)
                for playlist in mpd_slice["playlists"]:
                    playlist_id = playlist[self.RAW_PLAYLIST_ID_KEY]
                    playlist_timestamp = playlist[self.RAW_TIMESTAMP_KEY]
                    # Get songs in playlist
                    playlist_tracks = self._process_playlist(playlist)
                    for track in playlist_tracks:
                        dataset += [{self.SPOTIFY_SESSION_ID: playlist_id,
                                     self._SPOTIFY_TIME_COLUMN: playlist_timestamp,
                                     self.SPOTIFY_ITEM_ID: track.name,
                                     self.SPOTIFY_ALBUM_NAME_KEY: track.album,
                                     self.SPOTIFY_ARTIST_NAME_KEY: track.artist}]

        # Write data to CSV
        spotify_dataframe = pd.DataFrame(data=dataset,
                                         # index=index,
                                         columns=[self.SPOTIFY_SESSION_ID, self._SPOTIFY_TIME_COLUMN,
                                                  self.SPOTIFY_ITEM_ID, self.SPOTIFY_ALBUM_NAME_KEY,
                                                  self.SPOTIFY_ARTIST_NAME_KEY]
                                         )
        # spotify_dataframe.index.name = self.SPOTIFY_SESSION_ID
        if not os.path.exists(output_file):
            output_file.parent.mkdir(parents=True, exist_ok=True)
        spotify_dataframe.to_csv(output_file, sep=self.delimiter, index=False)


class MelonConverter(CsvConverter):
    RAW_TRACKS_KEY = "songs"
    RAW_TIMESTAMP_KEY = "updt_date"
    RAW_PLAYLIST_ID_KEY = "id"
    # tags, plylst_title (opt.), like_cnt

    _MELON_TIME_COLUMN = "playlist_timestamp"
    MELON_SESSION_ID = "playlist_id"

    MELON_ITEM_ID = "track_name"
    MELON_ALBUM_NAME_KEY = "album_name"
    MELON_ARTIST_NAME_KEY = "artist_name"
    MELON_GENRE_KEY = "genre"

    # song_gn_dtl_basket (subgenres), issue_date

    def __init__(self, delimiter="\t"):
        self.delimiter = delimiter

    def apply(self, input_dir: Path, output_file: Path):
        dataset: List[List[Any]] = []
        trackdict: Dict[Track] = {}
        filenames = os.listdir(input_dir)
        f = open(input_dir.joinpath("song_meta.json"))
        js = f.read()
        f.close()
        tracks = json.loads(js)
        for song in tracks:
            track_name: str = song["song_name"]
            album_name: str = song[self.MELON_ALBUM_NAME_KEY]
            artist_name: str = "|".join(song["artist_name_basket"])
            genre: str = "|".join(song["song_gn_gnr_basket"])
            trackdict[song["id"]] = Track(name=track_name, album=album_name, artist=artist_name, genre=genre)
        for filename in tqdm(sorted(filenames), desc=f"Process playlists in file"):
            if filename in ("train.json", "val.json", "test.json"):
                file_path: Path = input_dir.joinpath(filename)
                f = open(file_path)
                js = f.read()
                f.close()
                mpd_slice = json.loads(js)
                for playlist in mpd_slice:
                    playlist_id = playlist[self.RAW_PLAYLIST_ID_KEY]
                    # playlist_timestamp = playlist[self.RAW_TIMESTAMP_KEY]
                    playlist_songs = playlist[self.RAW_TRACKS_KEY]
                    # Get songs in playlist
                    for track in playlist_songs:
                        song_name = trackdict[track]['name']
                        album_name = trackdict[track]['album']
                        artist_name = trackdict[track]['artist']
                        genre_name = trackdict[track]['genre']
                        if song_name and album_name and artist_name and genre_name:
                            dataset += [{self.MELON_SESSION_ID: playlist_id,
                                         # self._MELON_TIME_COLUMN: playlist_timestamp,
                                         self.MELON_ITEM_ID: song_name,
                                         self.MELON_ALBUM_NAME_KEY: album_name,
                                         self.MELON_ARTIST_NAME_KEY: artist_name,
                                         self.MELON_GENRE_KEY: genre_name}]

        # Write data to CSV
        spotify_dataframe = pd.DataFrame(data=dataset,
                                         columns=[self.MELON_SESSION_ID, self.MELON_ITEM_ID, self.MELON_ALBUM_NAME_KEY,
                                                  self.MELON_ARTIST_NAME_KEY, self.MELON_GENRE_KEY]
                                         )
        if not os.path.exists(output_file):
            output_file.parent.mkdir(parents=True, exist_ok=True)
        spotify_dataframe.to_csv(output_file, sep=self.delimiter, index=False)


class CoveoConverter(CsvConverter):

    def __init__(self, end_of_train, end_of_validation, min_item_feedback, min_sequence_length, include_pageviews,
                 prefix, search_sessions_only = False, delimiter: str = "\t"):
        self.end_of_train = end_of_train
        self.end_of_validation = end_of_validation
        self.min_item_feedback = min_item_feedback
        self.min_sequence_length = min_sequence_length
        self.include_pageviews = include_pageviews
        self.delimiter = delimiter
        self.prefix = prefix
        self.search_list_page = prefix == "coveo-extended"
        self.search_sessions_only = search_sessions_only

    def apply(self, input_dir: Path, output_file: Path):
        output_dir = output_file.parent
        browsing_train, search_train, sku_to_content = self._load_raw_files(input_dir)

        self._convert_vectors_to_lists(search_train)
        self._convert_vectors_to_arrays(sku_to_content)

        # self._remove_duplicates(browsing_train)
        # browsing_train = self._add_search_clicks(browsing_train, search_train)
        browsing_train = self._handle_pageviews(browsing_train)

        full_dataset = self._merge_tables(browsing_train, sku_to_content)
        full_dataset = self._apply_min_item_feedback(full_dataset)
        self._fill_nan_values(full_dataset)

        desc_vector_dict = self._create_desc_vector_dict(sku_to_content)
        img_vector_dict = self._create_img_vector_dict(sku_to_content)

        search_train = self._prepare_search_list_pages(search_train, sku_to_content)

        full_dataset["item_id_type"] = 1
        test, train, validation = self._create_split(full_dataset, search=search_train)

        self._fill_nan_values(test)
        self._fill_nan_values(train)
        self._fill_nan_values(validation)

        if not os.path.exists(output_file):
            output_file.parent.mkdir(parents=True, exist_ok=True)
        self._export_files(desc_vector_dict, img_vector_dict, output_dir, test, train, validation, prefix=self.prefix)

    def _prepare_search_list_pages(self, search_clicks, sku_to_content):
        search_clicks = search_clicks[['session_id_hash', 'product_skus_hash', 'server_timestamp_epoch_ms']].copy()
        search_clicks.drop_duplicates(inplace=True)
        search_clicks = search_clicks[search_clicks['product_skus_hash'].notnull()]
        search_clicks["product_skus_hash"] = search_clicks["product_skus_hash"].str.replace('\[|\]|\'', '')
        search_clicks["product_skus_hash"] = search_clicks["product_skus_hash"].str.split(",")
        search_clicks = search_clicks.explode("product_skus_hash")
        category_dict = sku_to_content[["product_sku_hash", "category_hash"]].set_index("product_sku_hash").to_dict()[
            "category_hash"]
        search_clicks["category_hash"] = search_clicks["product_skus_hash"].map(category_dict)
        search_clicks.dropna()
        search_clicks = search_clicks[["session_id_hash", "server_timestamp_epoch_ms", 'category_hash']]
        search_clicks.assign(category_hash=search_clicks['category_hash'].str.split('/')).explode('category_hash')
        search_clicks = self.get_list_page_categories(search_clicks)
        search_clicks["item_id_type"] = 0
        search_clicks["event_type"] = "search"
        search_clicks["product_action"] = "search"
        search_clicks["product_sku_hash"] = "SEARCHLIST"
        search_clicks["hashed_url"] = "SEARCHLIST"
        search_clicks["price_bucket"] = "0"
        search_clicks['server_timestamp_epoch_ms'] = search_clicks[['server_timestamp_epoch_ms']].astype(int)

        return search_clicks

    def get_list_page_categories(self, df):
        def concat_values(df):
            return df['category_hash'].str.cat(sep='/')

        counts = df.groupby(['session_id_hash', 'server_timestamp_epoch_ms'])['category_hash'].value_counts().reset_index(
            name='count')
        counts = counts.groupby(["session_id_hash", "server_timestamp_epoch_ms"]).apply(
            lambda x: x.nlargest(5, 'count')).reset_index(drop=True)
        result = counts.groupby(["session_id_hash", "server_timestamp_epoch_ms"]).apply(concat_values).reset_index(
            name='category_hash')
        return result

    def _load_raw_files(self, input_dir):
        browsing_train = pd.read_csv(os.path.join(input_dir, "browsing_train.csv"), header=0)
        search_train = pd.read_csv(os.path.join(input_dir, "search_train.csv"), header=0)
        sku_to_content = pd.read_csv(os.path.join(input_dir, "sku_to_content.csv"), header=0)
        return browsing_train, search_train, sku_to_content

    def _convert_vectors_to_lists(self, search_train):
        search_train['clicked_skus_hash'] = search_train['clicked_skus_hash'].apply(self._convert_str_to_list)

    def _convert_str_to_list(self, x):
        if pd.isnull(x):
            return x
        return ast.literal_eval(x)

    def _convert_vectors_to_arrays(self, sku_to_content):
        sku_to_content['description_vector'] = sku_to_content['description_vector'].apply(self._convert_str_to_pdarray)
        sku_to_content['image_vector'] = sku_to_content['image_vector'].apply(self._convert_str_to_pdarray)

    def _convert_str_to_pdarray(self, x):
        if pd.isnull(x):
            return x
        list_x = ast.literal_eval(x)
        return pd.array(data=list_x, dtype=float)

    def _remove_duplicates(self, browsing_train):
        browsing_train.drop_duplicates(inplace=True)
        # Remove indices of 'pageview' interactions from duplicated events where an interaction generate a detail and a pageview event
        tmp = browsing_train[(browsing_train.event_type == 'pageview') & (
            browsing_train.duplicated(['session_id_hash', 'server_timestamp_epoch_ms'], keep="first"))]
        browsing_train.drop(tmp.index, inplace=True)
        tmp2 = browsing_train[(browsing_train.event_type == 'pageview') & (
            browsing_train.duplicated(['session_id_hash', 'server_timestamp_epoch_ms'], keep="last"))]
        browsing_train.drop(tmp2.index, inplace=True)

    def _add_search_clicks(self, browsing_train, search_train):
        search_clicks = self._extract_search_clicks(search_train)
        browsing_train = pd.concat([browsing_train, search_clicks])
        return browsing_train

    def _extract_search_clicks(self, search_train):
        search_clicks = search_train[['session_id_hash', 'clicked_skus_hash', 'server_timestamp_epoch_ms']].copy()
        search_clicks['event_type'] = 'event_product'
        search_clicks['product_action'] = 'search'
        search_clicks = search_clicks[search_clicks['clicked_skus_hash'].notnull()]
        search_clicks = self._unstack_list_of_clicked_items_to_multiple_rows(search_clicks)
        search_clicks['hashed_url'] = search_clicks['product_sku_hash']
        # duplicates could indicate interest but are removed here
        search_clicks.drop_duplicates(inplace=True)
        return search_clicks

    def _unstack_list_of_clicked_items_to_multiple_rows(self, search_clicks):
        lst_col = 'clicked_skus_hash'
        search_clicks = pd.DataFrame({
            col: np.repeat(search_clicks[col].values, search_clicks[lst_col].str.len()) for col in
            search_clicks.columns.difference([lst_col])}).assign(
            **{lst_col: np.concatenate(search_clicks[lst_col].values)})[search_clicks.columns.tolist()]
        search_clicks.columns = ['session_id_hash', 'product_sku_hash', 'server_timestamp_epoch_ms',
                                 'event_type', 'product_action']
        return search_clicks

    def _handle_pageviews(self, browsing_train):
        if self.include_pageviews:
            browsing_train['product_sku_hash'] = browsing_train['product_sku_hash'].fillna(browsing_train['hashed_url'])
        else:
            browsing_train = browsing_train[browsing_train['product_sku_hash'].notnull()]
        return browsing_train

    def _merge_tables(self, browsing_train, sku_to_content):
        full_dataset = pd.merge(browsing_train, sku_to_content, on='product_sku_hash', how='left')
        full_dataset.drop(columns=['description_vector', 'image_vector'], inplace=True)
        full_dataset.sort_values(['session_id_hash', 'server_timestamp_epoch_ms'], inplace=True)
        return full_dataset

    def _apply_min_item_feedback(self, full_dataset):
        aggregated = full_dataset[full_dataset['event_type'] == 'event_product'].groupby(['product_sku_hash']).size()
        filtered = aggregated.apply(lambda v: v >= self.min_item_feedback)
        filtered = filtered.reset_index()
        filtered.columns = ['product_sku_hash', 'item_feedback_bool']
        ids = filtered[filtered['item_feedback_bool'] == False]['product_sku_hash'].tolist()
        full_dataset = full_dataset[~full_dataset['product_sku_hash'].isin(ids)].copy()
        return full_dataset

    def _fill_nan_values(self, full_dataset):
        full_dataset.fillna(value={"product_action": "view", "price_bucket": "missing", "category_hash": "missing"},
                            inplace=True)

    def _create_desc_vector_dict(self, sku_to_content):
        desc_vector_dict = sku_to_content[['product_sku_hash', 'description_vector']]
        desc_vector_dict = desc_vector_dict[desc_vector_dict['description_vector'].notnull()]
        return desc_vector_dict

    def _create_img_vector_dict(self, sku_to_content):
        img_vector_dict = sku_to_content[['product_sku_hash', 'image_vector']]
        img_vector_dict = img_vector_dict[img_vector_dict['image_vector'].notnull()]
        return img_vector_dict

    def _create_split(self, full_dataset, search):

        full_dataset["category_product_id"]= full_dataset["product_sku_hash"]
        full_dataset.sort_values(['server_timestamp_epoch_ms'], inplace=True)
        train = full_dataset.loc[(full_dataset['server_timestamp_epoch_ms'] <= self.end_of_train)].copy()
        validation = full_dataset.loc[
            (full_dataset['server_timestamp_epoch_ms'] <= self.end_of_validation) & (
                    full_dataset['server_timestamp_epoch_ms'] > self.end_of_train)].copy()
        test = full_dataset.loc[(full_dataset['server_timestamp_epoch_ms'] > self.end_of_validation)].copy()

        train = self._apply_min_sequence_length(train)
        validation = self._apply_min_sequence_length(validation)
        test = self._apply_min_sequence_length(test)


        search["category_product_id"] = search["category_hash"]
        search_train = search.loc[(search['server_timestamp_epoch_ms'] <= self.end_of_train)].copy()
        search_validation = search.loc[
            (search['server_timestamp_epoch_ms'] <= self.end_of_validation) & (
                    search['server_timestamp_epoch_ms'] > self.end_of_train)].copy()
        search_test = search.loc[(search['server_timestamp_epoch_ms'] > self.end_of_validation)].copy()

        # Add search pages if necessary
        train = self._filtered_concat(train, search_train, self.search_sessions_only)
        validation = self._filtered_concat(validation, search_validation, self.search_sessions_only)
        test = self._filtered_concat(test, search_test, self.search_sessions_only)

        train.sort_values(['session_id_hash', 'server_timestamp_epoch_ms'], inplace=True)
        validation.sort_values(['session_id_hash', 'server_timestamp_epoch_ms'], inplace=True)
        test.sort_values(['session_id_hash', 'server_timestamp_epoch_ms'], inplace=True)

        return test, train, validation

    def _filtered_concat(self, dataframe, search, search_sessions_only):
        search = search[search['session_id_hash'].isin(dataframe['session_id_hash'])]
        if search_sessions_only:
            dataframe = dataframe[dataframe['session_id_hash'].isin(dataframe['session_id_hash'])]
        if self.search_list_page:
            return pd.concat([dataframe,search])
        return dataframe

    def _apply_min_sequence_length(self, dataset):
        aggregated = dataset.groupby(['session_id_hash']).size()
        filtered = aggregated.apply(lambda v: v >= self.min_sequence_length)
        filtered = filtered.reset_index()
        filtered.columns = ['session_id_hash', 'min_sequence_bool']
        ids = filtered[filtered['min_sequence_bool']]['session_id_hash'].tolist()
        dataset = dataset[dataset['session_id_hash'].isin(ids)].copy()
        return dataset

    def _export_files(self, desc_vector_dict, img_vector_dict, output_dir, test, train, validation, prefix="coveo"):
        os.makedirs(output_dir, exist_ok=True)

        train.to_csv(path_or_buf=os.path.join(output_dir, prefix + '.train.csv'), sep=self.delimiter, index=False)
        validation.to_csv(path_or_buf=os.path.join(output_dir, prefix + '.validation.csv'), sep=self.delimiter,
                          index=False)
        test.to_csv(path_or_buf=os.path.join(output_dir, prefix + '.test.csv'), sep=self.delimiter, index=False)
        # desc_vector_dict.to_csv(path_or_buf=os.path.join(output_dir, "desc_vector_dict.csv"), sep=self.delimiter,
        #                        index=False)
        # img_vector_dict.to_csv(path_or_buf=os.path.join(output_dir, "img_vector_dict.csv"), sep=self.delimiter,
        #                       index=False)


class HMConverter(CsvConverter):

    def __init__(self, end_of_train, end_of_validation, min_item_feedback, min_sequence_length, delimiter="\t"):
        self.end_of_train = end_of_train
        self.end_of_validation = end_of_validation
        self.min_item_feedback = min_item_feedback
        self.min_sequence_length = min_sequence_length
        self.delimiter = delimiter

    def apply(self, input_dir: Path, output_file: Path):
        output_dir = output_file.parent

        articles, customers, transactions = self._load_raw_files(input_dir)

        self._convert_dates_to_timestamps(transactions)
        self._drop_irrelevant_columns(articles)

        full_dataset = self._merge_tables(articles, customers, transactions)
        self._fill_nan_values_with_zero(full_dataset)
        full_dataset = self._apply_min_item_feedback(full_dataset)

        test, train, validation = self._create_split(full_dataset)
        if not os.path.exists(output_file):
            output_file.parent.mkdir(parents=True, exist_ok=True)
        self._export_split_files(output_dir, test, train, validation)

    def _load_raw_files(self, input_dir):
        articles = pd.read_csv(os.path.join(input_dir, "articles.csv"))
        customers = pd.read_csv(os.path.join(input_dir, "customers.csv"))
        transactions = pd.read_csv(os.path.join(input_dir, "transactions_train.csv"))
        return articles, customers, transactions

    def _convert_dates_to_timestamps(self, transactions):
        transactions['t_dat'] = transactions['t_dat'].apply(self._datetime_to_timestamp)

    def _datetime_to_timestamp(self, date):
        return time.mktime(datetime.strptime(date, "%Y-%m-%d").timetuple())

    def _drop_irrelevant_columns(self, articles):
        articles.drop(inplace=True,
                      columns=['product_code', 'product_type_no', 'graphical_appearance_no', 'colour_group_code',
                               'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no', 'index_code',
                               'index_group_no', 'section_no', 'garment_group_no'])

    def _merge_tables(self, articles, customers, transactions):
        full_dataset = pd.merge(transactions, customers, on='customer_id', how='left')
        full_dataset = pd.merge(full_dataset, articles, on='article_id', how='left')
        full_dataset.sort_values(['customer_id', 't_dat'], inplace=True)
        return full_dataset

    def _fill_nan_values_with_zero(self, full_dataset):
        full_dataset.fillna(0, inplace=True)

    def _apply_min_item_feedback(self, full_dataset):
        aggregated = full_dataset.groupby(['article_id']).size()
        filtered = aggregated.apply(lambda v: v >= self.min_item_feedback)
        filtered = filtered.reset_index()
        filtered.columns = ['article_id', 'item_feedback_bool']
        ids = filtered[filtered['item_feedback_bool']]['article_id'].tolist()
        full_dataset = full_dataset[full_dataset['article_id'].isin(ids)].copy()
        return full_dataset

    def _create_split(self, full_dataset):
        full_dataset.sort_values(['t_dat'], inplace=True)
        train = full_dataset.loc[(full_dataset['t_dat'] <= self.end_of_train)].copy()
        validation = full_dataset.loc[
            (full_dataset['t_dat'] <= self.end_of_validation) & (full_dataset['t_dat'] > self.end_of_train)].copy()
        test = full_dataset.loc[(full_dataset['t_dat'] > self.end_of_validation)].copy()

        train = self._apply_min_sequence_length(train)
        validation = self._apply_min_sequence_length(validation)
        test = self._apply_min_sequence_length(test)

        train.sort_values(['customer_id', 't_dat'], inplace=True)
        validation.sort_values(['customer_id', 't_dat'], inplace=True)
        test.sort_values(['customer_id', 't_dat'], inplace=True)
        return test, train, validation

    def _apply_min_sequence_length(self, dataset):
        aggregated = dataset.groupby(['customer_id']).size()
        filtered = aggregated.apply(lambda v: v >= self.min_sequence_length)
        filtered = filtered.reset_index()
        filtered.columns = ['customer_id', 'min_sequence_bool']
        ids = filtered[filtered['min_sequence_bool']]['customer_id'].tolist()
        dataset = dataset[dataset['customer_id'].isin(ids)].copy()
        return dataset

    def _export_split_files(self, output_dir, test, train, validation):
        train.to_csv(path_or_buf=os.path.join(output_dir, 'hm.train.csv'), sep=self.delimiter, index=False)
        validation.to_csv(path_or_buf=os.path.join(output_dir, 'hm.validation.csv'), sep=self.delimiter, index=False)
        test.to_csv(path_or_buf=os.path.join(output_dir, 'hm.test.csv'), sep=self.delimiter, index=False)


class ExampleConverter(CsvConverter):

    def __init__(self):
        pass

    def apply(self, input_dir: Path, output_file: Path):
        # We assume `input_dir` to be the path to the raw csv file.
        shutil.copy(input_dir, output_file)
