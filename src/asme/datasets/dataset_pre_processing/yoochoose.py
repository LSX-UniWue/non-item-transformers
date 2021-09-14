import os
from pathlib import Path
import numpy as np
import pandas as pd
import datetime as dt


FULL_TRAIN_SET = "full_training_set"
TRAIN_SET = "training_set"
VALIDATION_SET = "validation_set"
TEST_SET = "test_set"

YOOCHOOSE_SESSION_ID_KEY = "SessionId"
YOOCHOOSE_ITEM_ID_KEY = "ItemId"
YOOCHOOSE_CLICKS_FILE_NAME = "yoochoose-clicks"
YOOCHOOSE_BUYS_FILE_NAME = "yoochoose-buys"


def pre_process_yoochoose_dataset(path_to_original_data: Path,
                                  output_dir_path: Path,
                                  file_name:str
                                  ) -> Path:
    """
    Perform pre-processing for yoochoose data set as specified by Hidasi et al. 2016. Code adapted from
    https://github.com/hidasib/GRU4Rec/blob/master/examples/rsc15/preprocess.py.

    :param path_to_original_data: path to clicks.dat file
    :param output_dir_path: output dir for pre-processed csv
    :return: Stores preprocessed data at output_dir
    """
    data = pd.read_csv(path_to_original_data.joinpath(file_name+'.dat'),
                       sep=',',
                       header=None,
                       usecols=[0, 1, 2],
                       dtype={0: np.int32, 1: str, 2: np.int64},
                       names=['SessionId', 'TimeStr', 'ItemId'])

    data['Time'] = data.TimeStr.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())
    session_lengths = data.groupby(YOOCHOOSE_SESSION_ID_KEY).size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths > 1].index)]
    item_supports = data.groupby(YOOCHOOSE_ITEM_ID_KEY).size()
    data = data[np.in1d(data.ItemId, item_supports[item_supports >= 5].index)]
    session_lengths = data.groupby(YOOCHOOSE_SESSION_ID_KEY).size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths >= 2].index)]

    if not os.path.exists(output_dir_path):
        output_dir_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir_path.joinpath(file_name+'.csv')
    data = data.sort_values(YOOCHOOSE_SESSION_ID_KEY)
    data.to_csv(path_or_buf=output_file_path)
    return output_file_path

