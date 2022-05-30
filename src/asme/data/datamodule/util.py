import enum
import math
import os
from pathlib import Path
from typing import List

import pandas as pd
import requests
from tqdm import tqdm

TRAIN_KEY = "train"
VALIDATION_KEY = "validation"
TESTING_KEY = "test"


class SplitNames(enum.Enum):
    train = TRAIN_KEY
    validation = VALIDATION_KEY
    test = TESTING_KEY

    def __str__(self):
        return self.value


class TrainValidationTestSplitIndices:
    def __init__(self, train_indices: List[int], validation_indices: List[int], test_indices: List[int]):
        self.train_indices = train_indices
        self.validation_indices = validation_indices
        self.test_indices = test_indices

    def items(self):
        return [(SplitNames.train, self.train_indices),
                (SplitNames.validation, self.validation_indices),
                (SplitNames.test, self.test_indices)]

    def get(self, split_name: SplitNames):
        if split_name == SplitNames.train:
            return self.train_indices
        elif split_name == SplitNames.validation:
            return self.validation_indices
        elif split_name == SplitNames.test:
            return self.test_indices
        else:
            raise Exception(f"Provided invalid key '{split_name}' to TrainValidationTestSplitIndices.")


def download_dataset(url: str,
                     download_dir: Path
                     ) -> Path:
    """
    Downloads file if it doesn't already exists
    :param url:
    :param download_dir: the download dir
    :return:
    """
    filename = url.split("/")[-1]
    os.makedirs(download_dir, exist_ok=True)
    filepath = download_dir / filename
    if not os.path.exists(filepath):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(filepath, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
            progress_bar.close()
    else:
        print("File {} already downloaded".format(filepath))

    return filepath


def read_csv(dataset_dir: Path,
             file: str,
             file_type: str,
             sep: str,
             header: bool = None,
             encoding: str = "utf-8"
             ) -> pd.DataFrame:
    file_path = dataset_dir / f"{file}{file_type}"
    return pd.read_csv(file_path, sep=sep, header=header, engine="python", encoding=encoding)

def read_json(dataset_dir: Path,
             file: str,
             file_type: str,
             encoding: str = "utf-8"
             ) -> pd.DataFrame:
    file_path = dataset_dir / f"{file}{file_type}"
    return pd.read_json(file_path, engine="python", encoding=encoding)


def approx_equal(x: float, y: float, eps = 1e-9) -> bool:
    return math.fabs(x - y) < eps
