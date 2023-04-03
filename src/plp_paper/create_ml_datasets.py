import os
from pathlib import Path

import pandas as pd
import typer

from asme.data.datamodule.converters import CsvConverter
from asme.data.datamodule.util import read_csv
from asme.core.utils.run_utils import load_config, create_container
import typer
import torch

torch.multiprocessing.set_start_method('spawn', force=True)
app = typer.Typer()



import os
def create_extended_movielens_data(input_dir, output_dir, name, stage):
    file_type = ".csv"
    encoding = "latin-1"
    delimiter = "\t"
    item_df = read_csv(input_dir, f'{name}.{stage}', file_type, "\t", header=0, encoding=encoding)
    item_df["title_genres"] = item_df["title"]
    item_df["title_uid"] = item_df["title"]
    item_df["old_title"] = item_df["title"]
    item_df["item_id_type"] = 1
    page_df_mod = item_df.copy()
    page_df_mod["old_title"] = page_df_mod["title"]
    page_df_mod["title"] = "OVERVIEW-PAGE"
    page_df_mod["title_genres"] = page_df_mod["genres"]
    page_df_mod["title_uid"] = page_df_mod["userId"]
    page_df_mod["item_id_type"] = 0
    page_df_mod["rating"] = -1
    page_df_mod["year"] = 0
    item_df['original_order'] = item_df.groupby(['userId', 'timestamp']).cumcount() + 1
    page_df_mod['original_order'] = page_df_mod.groupby(['userId', 'timestamp']).cumcount() + 1
    item_df = pd.concat([item_df,page_df_mod], ignore_index=True)
    item_df = item_df.sort_values(["userId","timestamp","original_order","item_id_type"])
    if name == "ml-1m":
        item_df = item_df[['userId', 'rating', 'timestamp', 'gender', 'age',
                       'occupation', 'title', 'genres', 'year', 'user_all', 'title_genres', 'title_uid', 'item_id_type','zip']]
    else:
        item_df = item_df[['userId', 'rating', 'timestamp', 'title', 'genres', 'year', 'title_genres', 'title_uid', 'item_id_type']]
    os.makedirs(output_dir, exist_ok=True)
    item_df.to_csv(f'{output_dir}/{name+"-extended"}.{stage}{file_type}', sep=delimiter, index=False)

def create_original_data(config_file: Path = typer.Argument(..., help='the path to the config file', exists=True)) -> None:
    config_file_path = Path(config_file)
    config = load_config(config_file_path)
    #Creates the dataset:
    container = create_container(config)
    return container



@app.command()
def create(base_config: Path = typer.Argument(..., help='the path to the config file', exists=True),
           output_dir: Path = typer.Argument(..., help='the path to the config file', exists=False),
           name: str = typer.Option("ml-1m")):
    container = create_original_data(Path(base_config))
    input_dir = container._objects["current_split_path"]

    for stage in ["test","train","validation"]:
        create_extended_movielens_data(input_dir,output_dir,name,stage)


if __name__ == "__main__":
    app()
