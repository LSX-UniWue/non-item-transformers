import typer
from pathlib import Path

from datasets.app.data_set_commands import DEFAULT_SPECIAL_TOKENS
from datasets.vocabulary import create_vocabulary

app = typer.Typer()


@app.command()
def build(data_file_path: Path = typer.Argument(..., exists=True, help="path to the input file in CSV format"),
          session_index_path: Path = typer.Argument(..., exists=True, help="path to the session index file"),
          vocabulary_output_file_path: Path = typer.Argument(..., exists=True, help='Output path for vocab file'),
          item_header_name: str = typer.Argument(..., help="name of the column that contains the item id"),
          delimiter: str = typer.Option("\t", help="the delimiter used in the CSV file.")) -> None:
    """
    Builds item distribution for a given data set.

    :param data_file_path: CSV file containing original data
    :param session_index_path: index file belonging to the data file
    :param vocabulary_output_file_path: output path for vocabulary file
    :param item_header_name: Name of the item key in the data set, e.g, "ItemId"
    :param delimiter: delimiter used in data file
    :return: None, Side Effect: vocabulary for data file is written to vocabulary_output_file_path
    """
    create_vocabulary.create_token_vocabulary(column=item_header_name,
                                              data_file_path=data_file_path,
                                              session_index_path=session_index_path,
                                              vocabulary_output_file_path=vocabulary_output_file_path,
                                              custom_tokens=DEFAULT_SPECIAL_TOKENS,
                                              delimiter=delimiter)
