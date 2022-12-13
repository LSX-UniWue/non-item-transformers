import os
from pathlib import Path
from typing import List

from asme.core.init.context import Context
from asme.data.base.csv_index_builder import CsvSessionIndexer
from asme.data.base.reader import CsvDatasetIndex, CsvDatasetReader
from asme.data.datamodule.extractors import TargetPositionExtractor, SlidingWindowPositionExtractor
from asme.data.datamodule.preprocessing.action import PreprocessingAction, MAIN_FILE_KEY, DELIMITER_KEY, \
    SESSION_INDEX_KEY, OUTPUT_DIR_KEY, PREFIXES_KEY
from asme.data.datamodule.preprocessing.util import format_prefix
from asme.data.datasets.index_builder import SequencePositionIndexBuilder
from asme.data.datasets.sequence import ItemSessionParser, ItemSequenceDataset, PlainSequenceDataset, MetaInformation
from asme.data.utils.csv import create_indexed_header, read_csv_header


class CreateSessionIndex(PreprocessingAction):
    """
    Creates a session index for the provided session key.

    Required context parameters:
    - `MAIN_FILE_KEY` - The path to the main CSV file.
    - `OUTPUT_DIR_KEY` - The directory where the index should be saved.
    - `PREFIXES_KEY` - The prefixes to use when naming the index file.
    - `DELIMITER_KEY` - The delimiter that is used to separate columns in the main CSV file.

    Sets context parameters:
    - `SESSION_INDEX_KEY` - The path to the session index that was created in this step.
    """

    def __init__(self, session_key: List[str]):
        """
        :param session_key: The columns to be used to differentiate between sessions. If multiple keys are passed, they
                            are used lexicographically in the order of their appearance in the list.
        """
        self.session_key = session_key

    def name(self) -> str:
        return "Creating session index"

    def _run(self, context: Context) -> None:
        main_file = context.get(MAIN_FILE_KEY)
        session_index_path = self._get_session_index_path(context)
        delimiter = context.get(DELIMITER_KEY)
        csv_index = CsvSessionIndexer(delimiter=delimiter)
        csv_index.create(main_file, session_index_path, self.session_key)
        context.set(SESSION_INDEX_KEY, session_index_path, overwrite=True)

    def _dry_run(self, context: Context) -> None:
        context.set(SESSION_INDEX_KEY, self._get_session_index_path(context), overwrite=True)

    def dry_run_available(self, context: Context) -> bool:
        return os.path.exists(self._get_session_index_path(context))

    @staticmethod
    def _get_session_index_path(context: Context) -> Path:
        output_dir = context.get(OUTPUT_DIR_KEY)
        prefix = format_prefix(context.get(PREFIXES_KEY))
        return output_dir / f"{prefix}.session.idx"


class CreateNextItemIndex(PreprocessingAction):
    """
    Creates a next-item-index for each of the provided columns using the specified target position extractor.

    Required context parameters:
    - `MAIN_FILE_KEY` - The path to the main CSV file.
    - `OUTPUT_DIR_KEY` - The directory where the index should be saved.
    - `PREFIXES_KEY` - The prefixes to use when naming the index file.
    - `DELIMITER_KEY` - The delimiter that is used to separate columns in the main CSV file.
    - `SESSION_INDEX_KEY` - The path to the session index to consider to create the next-item index.

    Sets context parameters:
        None
    """

    def __init__(self, columns: List[MetaInformation], extractor: TargetPositionExtractor):
        """
        :param columns: A list of all columns for which a next-item-index should be generated.
        :param extractor: A target-position-extractor which is used to determine which elements of a session are used
                          as targets.
        """
        self.extractor = extractor
        self.columns = columns

    def name(self) -> str:
        return "Creating next item index"

    def _run(self, context: Context) -> None:
        main_file = context.get(MAIN_FILE_KEY)
        session_index_path = context.get(SESSION_INDEX_KEY)
        delimiter = context.get(DELIMITER_KEY)
        output_file = self._get_next_item_index_path(context)

        session_index = CsvDatasetIndex(session_index_path)
        reader = CsvDatasetReader(main_file, session_index)
        parser = ItemSessionParser(
            create_indexed_header(read_csv_header(main_file, delimiter)),
            self.columns,
            delimiter=delimiter
        )
        dataset = ItemSequenceDataset(PlainSequenceDataset(reader, parser))
        builder = SequencePositionIndexBuilder(target_positions_extractor=self.extractor)
        builder.build(dataset, output_file)

    def _dry_run(self, context: Context) -> None:
        pass

    def dry_run_available(self, context: Context) -> bool:
        return os.path.exists(self._get_next_item_index_path(context))

    @staticmethod
    def _get_next_item_index_path(context: Context) -> Path:
        output_dir = context.get(OUTPUT_DIR_KEY)
        prefix = format_prefix(context.get(PREFIXES_KEY))
        return output_dir / f"{prefix}.nextitem.idx"


class CreateSlidingWindowIndex(PreprocessingAction):
    """
    Creates a sliding-window index for each of the provided columns using the specified extractor.

    Required context parameters:
    - `MAIN_FILE_KEY` - The path to the main CSV file.
    - `OUTPUT_DIR_KEY` - The directory where the index should be saved.
    - `PREFIXES_KEY` - The prefixes to use when naming the index file.
    - `DELIMITER_KEY` - The delimiter that is used to separate columns in the main CSV file.
    - `SESSION_INDEX_KEY` - The path to the session index to consider to create the sliding-window index.

    Sets context parameters:
        None
    """

    def __init__(self,
                 columns: List[MetaInformation],
                 extractor: SlidingWindowPositionExtractor):
        """
        :param columns: A list of all columns for which a sliding-window index should be generated.
        :param extractor: A sliding-window-position-extractor which is used to determine which elements of a session are
                          used as targets.
        """
        self.extractor = extractor
        self.columns = columns

    def name(self) -> str:
        return f"Creating sliding window index with {self.extractor.window_size}/{self.extractor.session_end_offset}"

    def _run(self, context: Context) -> None:
        main_file = context.get(MAIN_FILE_KEY)
        session_index_path = context.get(SESSION_INDEX_KEY)
        delimiter = context.get(DELIMITER_KEY)

        output_file = self._get_sliding_window_index_path(context)

        session_index = CsvDatasetIndex(session_index_path)
        reader = CsvDatasetReader(main_file, session_index)
        parser = ItemSessionParser(
            create_indexed_header(read_csv_header(main_file, delimiter)),
            self.columns,
            delimiter=delimiter
        )
        dataset = ItemSequenceDataset(PlainSequenceDataset(reader, parser))
        builder = SequencePositionIndexBuilder(self.extractor)
        builder.build(dataset, output_file)

    def _dry_run(self, context: Context) -> None:
        pass

    def _get_sliding_window_index_path(self, context: Context) -> Path:
        output_dir = context.get(OUTPUT_DIR_KEY)
        prefix = format_prefix(context.get(PREFIXES_KEY))
        return output_dir / f"{prefix}.slidingwindow.{self.extractor.window_size}.idx"

    def dry_run_available(self, context: Context) -> bool:
        return os.path.exists(self._get_sliding_window_index_path(context))