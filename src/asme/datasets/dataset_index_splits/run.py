from pathlib import Path

from asme.datasets.app.data_set_commands import DEFAULT_SPECIAL_TOKENS
from asme.datasets.data_structures.dataset_metadata import DatasetMetadata
from asme.datasets.dataset_index_splits.conditional_split import create_sliding_window_index

if __name__ == '__main__':

    # base_path = Path('../dataset/ml-1m/')
    # dataset_metadata = DatasetMetadata(
    #     data_file_path=base_path / 'ml-1m.csv',
    #     session_key=[MOVIELENS_SESSION_KEY],
    #     item_header_name=MOVIELENS_ITEM_HEADER_NAME,
    #     delimiter=MOVIELENS_DELIMITER,
    #     special_tokens=DEFAULT_SPECIAL_TOKENS,
    #     stats_columns=None
    # )
    #
    # window_size = 5 + 3
    # result_file = base_path / 'loo' / f'ml-1m.train.slidingwindow.{window_size}.idx'
    # create_sliding_window_index(dataset_metadata, result_file, window_size, 2)
    base_path = Path('../../../../tests/example_dataset/ratio-0.8_0.1_0.1/')
    dataset_metadata = DatasetMetadata(
        data_file_path=base_path / 'example.train.csv',
        session_key=['session_id'],
        item_header_name='item_id',
        delimiter='\t',
        special_tokens=DEFAULT_SPECIAL_TOKENS,
        stats_columns=None
    )

    window_size = 2 + 1
    result_file = base_path / f'example.train.slidingwindow.{window_size}.idx'
    create_sliding_window_index(dataset_metadata, result_file, window_size, 0)
