import typer
from asme.datasets.app import data_set_commands, popularity_command, vocabulary_command, index_command
from asme.datasets.app import split_commands

"""
This file combines the Typer commands for the data set CLI.
Documentation can be found under docs/data_set_typer_app.md  
"""

app = typer.Typer()
# Register the Typer sub-commands
app.add_typer(data_set_commands.app, name="pre_process")
app.add_typer(index_command.app, name="index")
app.add_typer(split_commands.app, name="split")
app.add_typer(vocabulary_command.app, name="vocabulary")
app.add_typer(popularity_command.app, name="popularity")


def get_data_set_app() -> typer.Typer:
    return app


# Python main function to make Typer-App available as CLI.
if __name__ == "__main__":
    app()
