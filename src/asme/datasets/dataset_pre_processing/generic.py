from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Callable, Any

from asme.datasets.data_structures.dataset_metadata import DatasetMetadata
from asme.datasets.app import index_command
from asme.datasets.dataset_index_splits.conditional_split import all_remaining_positions, \
    create_conditional_index_using_extractor, run_loo_split
from asme.datasets.dataset_index_splits import strategy_split
from asme.datasets.vocabulary.create_vocabulary import create_token_vocabulary
from asme.datasets.popularity.build_popularity import build as build_popularity
from asme.datasets.data_structures.split_strategy import SplitStrategy


def generic_process_dataset(dataset_metadata: DatasetMetadata, split_strategies: List[SplitStrategy]) -> None:
    """
    Handles pre-processing, splitting, storing and indexing of a given session data sets.

    :param dataset_metadata: Data set metadata
    :return: Todo
    """
    # Index pre-processed data
    print("Indexing processed data...")
    index_command.index_csv(data_file_path=dataset_metadata.data_file_path,
                            index_file_path=dataset_metadata.session_index_path,
                            session_key=dataset_metadata.session_key,
                            delimiter=dataset_metadata.delimiter)

    for split_strategy in split_strategies:
        generic_strategy_split(dataset_metadata=dataset_metadata, split_strategy=split_strategy)

    # FIXME: add this to the split strategies
    print("Create Leave-One-Out split...")
    generic_leave_one_out_split(dataset_metadata=dataset_metadata)


def generic_strategy_split(dataset_metadata: DatasetMetadata, split_strategy: SplitStrategy):
    print(f'Create split using {type(split_strategy)} strategy')
    """
    Generating ratio_split, as well as corresponding vocabulary and popularity files for a specified data file and index
    :param dataset_metadata: Data set metadata
    :param minimum_session_length: the minimum acceptable session length
    :param split_strategy: Strategy that dictates the training, test, validation split
    :return: Todo
    """
    split_output_dir_path: Path = dataset_metadata.dataset_base_dir.joinpath(str(split_strategy))

    strategy_split.run_strategy_split(dataset_metadata=dataset_metadata, output_dir_path=split_output_dir_path,
                                      split_strategy=split_strategy)
    train_file_path: Path = split_output_dir_path / f"{dataset_metadata.file_prefix}.train.csv"
    train_session_index_file_path: Path = split_output_dir_path / f"{dataset_metadata.file_prefix}.train.session.idx"
    print("Build vocabulary and popularity for ratio split...")
    generic_create_vocabularies_and_popularities(dataset_metadata=dataset_metadata,
                                                 processed_data_file_path=train_file_path,
                                                 index_path=train_session_index_file_path,
                                                 output_dir_path=split_output_dir_path,
                                                 delimiter=dataset_metadata.delimiter)


def generic_leave_one_out_split(dataset_metadata: DatasetMetadata):
    """
    Creates a next item split, i.e., From every session with length k use sequence[0:k-2] for training,
    sequence[-2] for validation and sequence[-1] for testing. As well as

    :param dataset_metadata: Data set metadata
    :return: None, Side effect: Test and Validation indices are written
    """
    loo_output_dir_path: Path = dataset_metadata.dataset_base_dir / "loo"

    print("Create next item index...")
    # Create session index for training with all session having the form session[:k-2]
    training_session_index_file: Path = loo_output_dir_path / f"{dataset_metadata.file_prefix}.train.nextitem.idx"
    create_conditional_index_using_extractor(dataset_metadata=dataset_metadata,
                                             output_file_path=training_session_index_file,
                                             target_positions_extractor=all_remaining_positions)

    print("Create leave one out index...")
    run_loo_split(dataset_metadata=dataset_metadata,
                  output_dir_path=loo_output_dir_path)

    print("Write vocabularies and popularities...")
    generic_create_vocabularies_and_popularities(dataset_metadata=dataset_metadata,
                                                 processed_data_file_path=dataset_metadata.data_file_path,
                                                 index_path=dataset_metadata.session_index_path,
                                                 output_dir_path=loo_output_dir_path,
                                                 delimiter=dataset_metadata.delimiter)


def generic_create_vocabularies_and_popularities(dataset_metadata: DatasetMetadata,
                                                 processed_data_file_path: Path,
                                                 index_path: Path,
                                                 output_dir_path: Path,
                                                 delimiter: str,
                                                 strategy_function: Optional[Callable[[List[Any]], List[Any]]] = None):
    """
    Writes vocabularies and popularity for the specified columns in the index and data file.

    :param dataset_metadata: Data Set metadata
    :param processed_data_file_path: data file that the split should be created for
    :param index_path: session index belonging to the data file
    :param output_dir_path: output directory where the index files for the splits are written to
    :param delimiter: delimiter used in data file
    :param strategy_function: function selecting which items of a session are used in the vocabulary and popularity
    :return: None, Side effect: Popularities and vocabularies are written
    """
    file_prefix: str = dataset_metadata.file_prefix
    for column_name in tqdm(dataset_metadata.stats_columns,
                            desc="Create Vocabulary and Popularity for every Stats Column"):
        vocabulary_output_file_path: Path = output_dir_path / f"{file_prefix}.vocabulary.{column_name}.txt"
        popularity_output_file_path: Path = output_dir_path / f"{file_prefix}.popularity.{column_name}.txt"

        create_token_vocabulary(column=column_name,
                                data_file_path=processed_data_file_path,
                                session_index_path=index_path,
                                vocabulary_output_file_path=vocabulary_output_file_path,
                                custom_tokens=dataset_metadata.custom_tokens,
                                delimiter=delimiter,
                                strategy_function=strategy_function)
        build_popularity(data_file_path=processed_data_file_path,
                         session_index_path=index_path,
                         vocabulary_file_path=vocabulary_output_file_path,
                         output_file_path=popularity_output_file_path,
                         item_header_name=column_name,
                         delimiter=delimiter,
                         strategy_function=strategy_function)
