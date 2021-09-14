import typer
from pathlib import Path
from typing import List
from asme.datasets.dataset_index_splits import conditional_split, strategy_split
from asme.datasets.data_structures.dataset_metadata import DatasetMetadata
from asme.datasets.data_structures.split_strategy import SplitStrategy
from asme.datasets.dataset_index_splits import split_strategies_factory


app = typer.Typer()


@app.command()
def next_item(data_file_path: Path = typer.Argument(..., help="Data file in csv format"),
              session_index_path: Path = typer.Argument(..., help="Path to session index for the data file"),
              output_dir_path: Path = typer.Argument(..., help="path that the splits should be written to"),
              delimiter: str = typer.Option('\t', help="Delimiter used in data file"),
              item_header: str = typer.Option('title', help="Dataset Key that the Item-IDs are stored under")):
    """
    Creates a next item split, i.e., From every session with length k use sequence[0:k-2] for training,
    sequence[-2] for validation and sequence[-1] for testing.

    :param data_file_path: data file that the split should be created for
    :param session_index_path: session index belonging to the data file
    :param output_dir_path: output directory where the index files for the splits are written to
    :param delimiter: delimiter used in data file
    :param item_header: data set key that the item-ids are stored under
    :return: None, Side effect: Test and Validation indices are written
    """
    dataset_metadata: DatasetMetadata = DatasetMetadata(
        data_file_path=data_file_path,
        session_index_path=session_index_path,
        session_key=None,
        delimiter=delimiter,
        item_header_name=item_header
    )
    conditional_split.run_loo_split(dataset_metadata=dataset_metadata, output_dir_path=output_dir_path)


@app.command()
def ratios(
        data_file_path: Path = typer.Argument(..., help="Data file in csv format"),
        session_index_path: Path = typer.Argument(..., help="Path to session index for the data file"),
        output_dir_path: Path = typer.Argument(..., help="path that the splits should be written to"),
        session_key: List[str] = typer.Argument(..., help="session key"),
        item_header_name: str = typer.Argument(..., help="name of the column that contains the item id"),
        train_ratio: float = typer.Argument(0.9, help="a list of splits, e.g. train;0.9 valid;0.05 test;0.05"),
        validation_ratio: float = typer.Argument(0.05, help="a list of splits, e.g. train;0.9 valid;0.05 test;0.05"),
        testing_ratio: float = typer.Argument(0.05, help="a list of splits, e.g. train;0.9 valid;0.05 test;0.05"),
        delimiter: str = typer.Option('\t', help="Delimiter used in data file"),
        seed: int = typer.Option(123456, help="Seed for random sampling of splits")):
    """
    Creates a data set split based on ratios where a percentage of the sessions are used for training, validation, and
    testing.
    :param data_file_path: data file that the split should be created for
    :param session_index_path: session index belonging to the data file
    :param output_dir_path: output directory where the data and index files for the splits are written to
    :param session_key: Session key used to uniquely identify sessions
    :param item_header_name: data set key that the item-ids are stored under
    :param train_ratio: share of session used for training
    :param validation_ratio: share of session used for validation
    :param testing_ratio: share of session used for testing
    :param delimiter: delimiter used in data file
    :param seed: Seed for random sampling
    :return: None, Side effects: CSV Files for splits are written
     """
    dataset_metadata: DatasetMetadata = DatasetMetadata(data_file_path=data_file_path,
                                                        session_key=session_key,
                                                        item_header_name=item_header_name, delimiter=delimiter,
                                                        session_index_path=session_index_path)
    ratio_split_strategy: SplitStrategy = split_strategies_factory.get_ratio_strategy(train_ratio=train_ratio,
                                                                                      validation_ratio=validation_ratio,
                                                                                      test_ratio=testing_ratio,
                                                                                      seed=seed)
    strategy_split.run_strategy_split(dataset_metadata=dataset_metadata, output_dir_path=output_dir_path,
                                      split_strategy=ratio_split_strategy)


# Todo Find better name
@app.command()
def create_conditional_index(
        data_file_path: Path = typer.Argument(..., exists=True, help="path to the input file in CSV format"),
        session_index_path: Path = typer.Argument(..., exists=True, help="path to the session index file"),
        output_file_path: Path = typer.Argument(..., help="path to the output file"),
        item_header_name: str = typer.Argument(..., help="name of the column that contains the item id"),
        delimiter: str = typer.Option("\t", help="the delimiter used in the CSV file."),
        target_feature: str = typer.Option(None, help="the target column name to build the targets against"
                                                      "(default all next subsequences will be considered);"
                                                      "the target must be a boolean feature"),
        min_sequence_length: int = typer.Option(0, help="the minimum sequence length to consider the target feature")
        ) -> None:
    """
    TODO: add documentation

    :param data_file_path:
    :param session_index_path:
    :param output_file_path:
    :param item_header_name:
    :param delimiter:
    :param target_feature:
    :param min_sequence_length
    :return:
    """
    dataset_metadata: DatasetMetadata = DatasetMetadata(
        data_file_path=data_file_path,
        session_index_path=session_index_path,
        delimiter=delimiter,
        item_header_name=item_header_name,
        session_key=None
    )
    conditional_split.create_conditional_index(dataset_metadata=dataset_metadata, output_file_path=output_file_path,
                                               target_feature=target_feature, min_sequence_length=min_sequence_length)
