import os
import shutil
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Any, Dict
from tqdm import tqdm

import numpy as np
import pandas as pd

from asme.data.datamodule.util import read_csv, read_json
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
        movies_df["year"] = movies_df["title"].str.rsplit(r"(", 1).apply(lambda x: x[1].rsplit(r")")[0]).astype(int)
        users_df = read_csv(location, "users", file_type, sep, header, encoding=encoding)
        users_df.columns = [Movielens1MConverter.RATING_USER_COLUMN_NAME, 'gender', 'age', 'occupation', 'zip']
        ratings_df = pd.merge(ratings_df, users_df)

        merged_df = pd.merge(ratings_df, movies_df).sort_values(
            by=[Movielens1MConverter.RATING_USER_COLUMN_NAME, Movielens1MConverter.RATING_TIMESTAMP_COLUMN_NAME])

        os.makedirs(output_file.parent, exist_ok=True)
        merged_df["user_all"] = merged_df["gender"].astype(str)+"|"+merged_df["age"].astype(str)+"age|"+merged_df["occupation"].astype(str)+"occupation"
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
                                         #index=index,
                                         columns=[self.SPOTIFY_SESSION_ID, self._SPOTIFY_TIME_COLUMN, self.SPOTIFY_ITEM_ID, self.SPOTIFY_ALBUM_NAME_KEY, self.SPOTIFY_ARTIST_NAME_KEY]
                                        )
        #spotify_dataframe.index.name = self.SPOTIFY_SESSION_ID
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
                                         #self._MELON_TIME_COLUMN: playlist_timestamp,
                                         self.MELON_ITEM_ID: song_name,
                                         self.MELON_ALBUM_NAME_KEY: album_name,
                                         self.MELON_ARTIST_NAME_KEY: artist_name,
                                         self.MELON_GENRE_KEY: genre_name}]

        # Write data to CSV
        spotify_dataframe = pd.DataFrame(data=dataset,
                                         columns=[self.MELON_SESSION_ID, self.MELON_ITEM_ID, self.MELON_ALBUM_NAME_KEY, self.MELON_ARTIST_NAME_KEY, self.MELON_GENRE_KEY]
                                         )
        if not os.path.exists(output_file):
            output_file.parent.mkdir(parents=True, exist_ok=True)
        spotify_dataframe.to_csv(output_file, sep=self.delimiter, index=False)


class ExampleConverter(CsvConverter):

    def __init__(self):
        pass

    def apply(self, input_dir: Path, output_file: Path):
        # We assume `input_dir` to be the path to the raw csv file.
        shutil.copy(input_dir, output_file)
