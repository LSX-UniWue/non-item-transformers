import typer
from pathlib import Path
from asme.datasets.popularity import build_popularity

app = typer.Typer()


@app.command()
def build(
        data_file_path: Path = typer.Argument(..., exists=True, help="path to the input file in CSV format"),
        session_index_path: Path = typer.Argument(..., exists=True, help="path to the session index file"),
        vocabulary_file_path: Path = typer.Argument(..., exists=True, help='path to the vocab file'),
        output_file_path: Path = typer.Argument(..., help="path to the popularity output file"),
        item_header_name: str = typer.Argument(..., help="name of the column that contains the item id"),
        min_session_length: int = typer.Option(2, help="the minimum acceptable session length"),
        delimiter: str = typer.Option("\t", help="the delimiter used in the CSV file.")) -> None:
    """
    Wrapper Command for the building of a item distribution.

    :param data_file_path: CSV file containing original data
    :param session_index_path: index file belonging to the data file
    :param vocabulary_file_path: vocabulary file belonging to the data file
    :param output_file_path: output file where the popularity should be written to
    :param item_header_name: Name of the item key in the data set, e.g, "ItemId"
    :param min_session_length: minimum session length determining which sessions should be used
    :param delimiter: delimiter used in data file
    :return: None, Side Effect popularity distribution is written
    """
    build_popularity.build(data_file_path=data_file_path, session_index_path=session_index_path,
                           vocabulary_file_path=vocabulary_file_path, output_file_path=output_file_path,
                           item_header_name=item_header_name, delimiter=delimiter)
