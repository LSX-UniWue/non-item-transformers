import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any, Optional

import pandas as pd

from asme.core.init.context import Context
from asme.data.datamodule.converters import CsvConverter
from asme.data.datamodule.preprocessing.action import PreprocessingAction, INPUT_DIR_KEY, MAIN_FILE_KEY, \
    OUTPUT_DIR_KEY, PREFIXES_KEY, DELIMITER_KEY
from asme.data.datamodule.preprocessing.util import format_prefix


class UseExistingCsv(PreprocessingAction):
    """
    Registers a pre-processed CSV file in context.

    Required context parameters:
    - `INPUT_DIR_KEY` - the path to the pre-processed CSV file.

    Sets context parameters:
    - `MAIN_FILE_KEY` - the path to the pre-processed CSV file.
    """

    def name(self) -> str:
        return "Use existing CSV file"

    def _dry_run(self, context: Context) -> None:
        if not context.has_path(INPUT_DIR_KEY):
            raise Exception(f"A pre-processed CSV file must be present in context at: '{INPUT_DIR_KEY}'")

        context.set(MAIN_FILE_KEY, context.get(INPUT_DIR_KEY))

    def _run(self, context: Context) -> None:
        self._dry_run(context)

    def dry_run_available(self, context: Context) -> bool:
        return True


class ConvertToCsv(PreprocessingAction):
    """
    Applies a conversion to the CSV format on a dataset.

    Required context parameters:
    - `INPUT_DIR_KEY` - the path to a dataset file
    - `OUTPUT_DIRECTORY_KEY` - the directory where the final converted CSV file will be placed
    - `PREFIXES_KEY` - the prefixes used to generate the name of the final CSV file.

    Sets context parameters:
    - `MAIN_FILE_KEY` - the path to the pre-processed CSV file.
    """

    def __init__(self, converter: CsvConverter):
        """
        :param converter: The converter that is used to create an authoritative CSV file for the dataset.
        """
        self.converter = converter

    def name(self) -> str:
        return "Converting to CSV"

    def _run(self, context: Context) -> None:
        input_dir = context.get(INPUT_DIR_KEY)
        output_directory = context.get(OUTPUT_DIR_KEY)

        if not output_directory.exists():
            output_directory.mkdir(parents=True)

        output_file = self._get_output_file(context)
        self.converter(input_dir, output_file)
        context.set(MAIN_FILE_KEY, output_file)

    def _dry_run(self, context: Context) -> None:
        context.set(MAIN_FILE_KEY, self._get_output_file(context))

    def dry_run_available(self, context: Context) -> bool:
        return os.path.exists(self._get_output_file(context))

    @staticmethod
    def _get_output_file(context: Context) -> Path:
        output_directory = context.get(OUTPUT_DIR_KEY)
        filename = f"{format_prefix(context.get(PREFIXES_KEY))}-raw.csv"
        return output_directory / filename


class CopyMainFile(PreprocessingAction):

    def name(self) -> str:
        return "Copying current main file to final location."

    def _run(self, context: Context) -> None:
        current_main_file_path = context.get(MAIN_FILE_KEY)
        new_main_file_path = self._get_final_location(context)

        # Ensure the directories for the new main file path exist
        new_main_file_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(current_main_file_path, new_main_file_path)
        context.set(MAIN_FILE_KEY, self._get_final_location(context), overwrite=True)

    def _dry_run(self, context: Context) -> None:
        context.set(MAIN_FILE_KEY, self._get_final_location(context), overwrite=True)

    def dry_run_available(self, context: Context) -> bool:
        return os.path.exists(self._get_final_location(context))

    @staticmethod
    def _get_final_location(context: Context) -> Path:
        output_dir = context.get(OUTPUT_DIR_KEY)
        prefixes = context.get(PREFIXES_KEY)
        prefix = format_prefix(prefixes)
        return output_dir / f"{prefix}.csv"


class TransformCsv(PreprocessingAction):
    """
    Reads the current main CSV file and transforms it via the provided function. The main CSV file is overwritten with
    the transformed one.

    Required context parameters:
    - `MAIN_FILE_KEY` - The path to the main CSV file.
    - `DELIMITER_KEY` - The delimiter used to separate columns in the main CSV file.

    Sets context parameters:
    - `MAIN_FILE_KEY` - The path to the new main CSV file.
    """

    def __init__(self, suffix: str, transform: Callable[[pd.DataFrame], pd.DataFrame]):
        """
        :param transform: A function that accepts a data-frame, processes it in some manner and returns it afterwards.
        :param suffix: A suffix to append to the transformed file.
        """
        self.suffix = suffix
        self.transform = transform

    def name(self) -> str:
        return "Filtering CSV"

    def _run(self, context: Context) -> None:
        current_file = context.get(MAIN_FILE_KEY)
        delimiter = context.get(DELIMITER_KEY)
        output_file = self._get_output_file_name(context)
        df = pd.read_csv(current_file, delimiter=delimiter, index_col=False)
        filtered = self.transform(df)
        filtered.to_csv(output_file, sep=delimiter, index=False)
        context.set(MAIN_FILE_KEY, output_file, overwrite=True)

    def _dry_run(self, context: Context) -> None:
        output_file = self._get_output_file_name(context)
        context.set(MAIN_FILE_KEY, output_file, overwrite=True)

    def dry_run_available(self, context: Context) -> bool:
        return os.path.exists(self._get_output_file_name(context))

    def _get_output_file_name(self, context: Context) -> Path:
        current_file = context.get(MAIN_FILE_KEY)
        output_dir = current_file.parent
        name, extension = os.path.splitext(current_file.name)
        return output_dir / f"{name}-{self.suffix}{extension}"


@dataclass
class GroupedFilter:
    """
    Objects of this class hold all information necessary to aggregate a grouped dataframe and the filter based on
    aggregated values.

    :param aggregator: An aggregation function provided by pandas either as a string or a function reference, e.g "count".
    :param apply: The actual filter function which determines whether a row should by kept based on the aggregated values.
    :param aggregated_column: The name of the column to which the filter will be applied. If this is None, the first
                              column of the grouped dataframe is used.
    """
    aggregator: Any
    apply: Callable[[Any], bool]
    aggregated_column: Optional[str] = None


class GroupAndFilter(PreprocessingAction):
    """
    Groups a dataframe by a key, then aggregates and filters the grouped dataframe.

    Required context parameters:
    - `MAIN_FILE_KEY` - The path to the main CSV file.
    - `DELIMITER_KEY` - The delimiter used to separate columns in the main CSV file.

    Sets context parameters:
        None

    """

    def __init__(self, suffix: str, group_by: str, filter: GroupedFilter):
        """
        :param suffix: The suffix to add to the filtered file.
        :param group_by: The column the data-frame should be grouped by.
        :param filter: A grouped filter instance containing the necessary information to perform aggregation and
                       filtering.
        """

        self.suffix = suffix
        self.group_by = group_by
        self.filter = filter

        def apply_filter(d):
            if filter.aggregated_column is None:
                agg_column = d.axes[0][0]
            else:
                agg_column = filter.aggregated_column
            agg_value = d[agg_column]
            return filter.apply(agg_value)

        def filter_fn(df: pd.DataFrame) -> pd.DataFrame:
            aggregated = df.groupby(self.group_by).agg(self.filter.aggregator)
            filtered = aggregated[aggregated.apply(apply_filter, axis=1)]
            filtered = filtered.reset_index()
            ids = filtered[self.group_by].unique().tolist()
            return df[df[group_by].isin(ids)]

        self.transform = TransformCsv(suffix, filter_fn)

    def name(self) -> str:
        return "Filtering sessions"

    def _run(self, context: Context) -> None:
        self.transform._run(context)

    def _dry_run(self, context: Context) -> None:
        self.transform._dry_run(context)

    def dry_run_available(self, context: Context) -> bool:
        return self.transform.dry_run_available(context)