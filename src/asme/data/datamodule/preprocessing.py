import copy
import math
import os
import random
import shutil
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Any, Optional, Dict

import pandas as pd

from asme.core.init.context import Context
from asme.core.tokenization.tokenizer import Tokenizer
from asme.core.tokenization.vocabulary import CSVVocabularyReaderWriter, Vocabulary
from asme.core.utils.logging import get_root_logger
from asme.data import RATIO_SPLIT_PATH_CONTEXT_KEY, LOO_SPLIT_PATH_CONTEXT_KEY
from asme.data.base.csv_index_builder import CsvSessionIndexer
from asme.data.base.reader import CsvDatasetIndex, CsvDatasetReader
from asme.data.datamodule.converters import CsvConverter
from asme.data.datamodule.extractors import TargetPositionExtractor, FixedOffsetPositionExtractor, \
    SlidingWindowPositionExtractor
from asme.data.datasets.index_builder import SequencePositionIndexBuilder
from asme.data.datasets.sequence import ItemSessionParser, ItemSequenceDataset, PlainSequenceDataset, MetaInformation
from asme.data.utils.csv import read_csv_header, create_indexed_header
from asme.datasets.data_structures.split_names import SplitNames
from asme.datasets.data_structures.train_validation_test_splits_indices import TrainValidationTestSplitIndices
from asme.datasets.vocabulary.create_vocabulary import create_token_vocabulary

# TODO (AD): check handling of prefixes, i think that is overengineered
MAIN_FILE_KEY = "main_file"
SESSION_INDEX_KEY = "session_index"
OUTPUT_DIR_KEY = "output_dir"
PREFIXES_KEY = "prefixes"
DELIMITER_KEY = "delimiter"
SEED_KEY = "seed"
INPUT_DIR_KEY = "input_dir"
SPLIT_BASE_DIRECTORY_PATH = "split_base_directory"
SPLIT_FILE_PREFIX = "split_file_prefix"
SPLIT_FILE_SUFFIX = "split_file_suffix"


def format_prefix(prefixes: List[str]) -> str:
    return ".".join(prefixes)


class PreprocessingAction:
    """
    Base class for all actions that are performed by the AsmeDatamodule during preprocessing of dataset.
    """
    @abstractmethod
    def name(self) -> str:
        """
        The name of the preprocessing action. Used for logging progress.
        """
        pass

    def apply(self, context: Context, force_execution: bool = False) -> None:
        """
        Applies the preprocessing action. It can rely on information that has been saved in the context by previous
        actions and store data itself.

        :param context: Context that is preserved between actions. Actions can use the information to find new files
                        on-the-fly and hand on new data to down-stream actions.
        :param force_execution: If set to True, this disables dry-runs and forces each action to generate all files,
                                regardless of whether they are already available.
        """
        if not self.dry_run_available(context) or force_execution:
            self._run(context)
        else:
            self._dry_run(context)

    @abstractmethod
    def _dry_run(self, context: Context) -> None:
        """
        This should not generate new files, etc. but only populates the context with the information it would usually
        place there.
        """
        pass

    @abstractmethod
    def _run(self, context: Context) -> None:
        """
        This method should actually perform the preprocessing action. It should also populate the context with the
        information need the for stages that consume artifacts of this action.
        """
        pass

    @abstractmethod
    def dry_run_available(self, context: Context) -> bool:
        """
        Indicates whether a stage is capable of skipping generation of files if they are already present. It needs to be
        able to populate the context with the necessary keys for later stages nonetheless.
        """
        pass

    def __call__(self, context: Context, force_execution: bool = False) -> None:
        self.apply(context, force_execution)


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
        ws = self.extractor.window_size + self.extractor.session_end_offset
        return output_dir / f"{prefix}.slidingwindow.{ws}.idx"

    def dry_run_available(self, context: Context) -> bool:
        return os.path.exists(self._get_sliding_window_index_path(context))


class CreateLeaveOneOutSplit(PreprocessingAction):
    """
    Creates a leave-one-out-split for the dataset using the provided column. Each supplied inner action is applied to 
    the split afterwards.
    
    Required context parameters:
    - `MAIN_FILE_KEY` - The path to the main CSV file.
    - `OUTPUT_DIR_KEY` - The directory where the index should be saved.
    - `PREFIXES_KEY` - The prefixes to use when naming the index file.
    - `DELIMITER_KEY` - The delimiter that is used to separate columns in the main CSV file.
    - `SESSION_INDEX_KEY` - The path to the session index to consider to create the leave-one-out split.
    
    Sets context parameters:
    - `LOO_SPLIT_PATH_CONTEXT_KEY` - The base path of the leave-one-out split.
    
    """
    def __init__(self, column: MetaInformation, training_target_offset=2, validation_target_offset=1, test_target_offset=0,
                 inner_actions: List[PreprocessingAction] = None):
        """
        This action allows to create a leave-one-out split for a dataset.

        :param column: The name of the column of the CSV file containing the items.
        :param training_target_offset:      The offset of the training target item, i.e. the distance from the end of
                                            the sequence. For instance, setting this to 2 will yield item n-2 as the
                                            target for a sequence of length n.
        :param validation_target_offset:    The offset of the validation target item, i.e. the distance from the end of
                                            the sequence. For instance, setting this to 1 will yield item n-1 as the
                                            target for a sequence of length n.
        :param test_target_offset:          The offset of the test target item, i.e. the distance from the end of the
                                            sequence. For instance, setting this to 0 will yield item n as the target
                                            for a sequence of length n.
        """
        self.inner_actions = [] if inner_actions is None else inner_actions
        self.column = column
        self.offsets = [training_target_offset, validation_target_offset, test_target_offset]

    def name(self) -> str:
        return "Creating Leave one out split"

    def _run(self, context: Context) -> None:
        output_dir = self._get_output_dir(context)
        main_file = context.get(MAIN_FILE_KEY)
        session_index_path = context.get(SESSION_INDEX_KEY)
        delimiter = context.get(DELIMITER_KEY)

        output_dir.mkdir(parents=True, exist_ok=True)
        # Save the path to the split in the context for the datamodule to use later
        context.set(LOO_SPLIT_PATH_CONTEXT_KEY, output_dir)

        session_index = CsvDatasetIndex(session_index_path)
        reader = CsvDatasetReader(main_file, session_index)
        parser = ItemSessionParser(
            create_indexed_header(read_csv_header(main_file, delimiter)),
            [self.column],
            delimiter=delimiter
        )
        dataset = ItemSequenceDataset(PlainSequenceDataset(reader, parser))
        for name, offset in zip([SplitNames.train, SplitNames.validation, SplitNames.test], self.offsets):
            prefix = format_prefix(context.get(PREFIXES_KEY) + [name.name])
            output_file = output_dir / f"{prefix}.loo.idx"
            builder = SequencePositionIndexBuilder(target_positions_extractor=FixedOffsetPositionExtractor(offset))
            builder.build(dataset, output_file)

        cloned = self._prepare_context_for_complete_split_actions(context)
        for action in self.inner_actions:
            action(cloned)

    def _dry_run(self, context: Context) -> None:
        context.set(LOO_SPLIT_PATH_CONTEXT_KEY, self._get_output_dir(context))

    def dry_run_available(self, context: Context) -> bool:
        cloned = self._prepare_context_for_complete_split_actions(context)
        for action in self.inner_actions:
            if not action.dry_run_available(cloned):
                return False

        return True

    def _prepare_context_for_complete_split_actions(self, context: Context) -> Context:
        output_dir = self._get_output_dir(context)
        cloned = copy.deepcopy(context)
        cloned.set(OUTPUT_DIR_KEY, output_dir, overwrite=True)
        return cloned

    @staticmethod
    def _get_output_dir(context: Context) -> Path:
        return context.get(OUTPUT_DIR_KEY) / "loo"


class CreateVocabulary(PreprocessingAction):
    """
    Creates a vocabulary file for each provided column.
    
    Required context parameters:
    - `MAIN_FILE_KEY` - The path to the main CSV file.
    - `OUTPUT_DIR_KEY` - The directory where the vocabulary should be saved.
    - `PREFIXES_KEY` - The prefixes to use when naming the vocabulary file.
    - `DELIMITER_KEY` - The delimiter that is used to separate columns in the main CSV file.
    - `SESSION_INDEX_KEY` - The path to the session index to consider to create the vocabulary.
    
    Sets context parameters:
        None
    """
    
    def __init__(self, columns: List[MetaInformation],
                 special_tokens: List[str] = None,
                 prefixes: List[str] = None):

        """
        :param columns: A list of all columns for which a vocabulary should be generated. Note that a column is skipped,
                        if its `run_tokenization` flag is set to false.
        :param special_tokens: A list of special tokens to artificially insert into
                               the vocabulary. By default ['<PAD>', '<UNK>', '<MASK>'] are inserted.
        :param prefixes: Allows to overwrite the prefixes used for naming the vocabulary files. If None is passed, the
                         prefixes are determined from the context.
        """
        self.columns = columns
        self.special_tokens = ['<PAD>', '<MASK>', '<UNK>'] if special_tokens is None else special_tokens
        self.prefixes = prefixes

    def name(self) -> str:
        return f"Creating Vocabulary for columns: {self.columns}."

    def _run(self, context: Context) -> None:
        main_file = context.get(MAIN_FILE_KEY)
        session_index_path = context.get(SESSION_INDEX_KEY)
        delimiter = context.get(DELIMITER_KEY)
        for column in self.columns:
            if not column.run_tokenization:
                get_root_logger().warning(f"Skipping vocabulary generation for '{column.column_name if column.column_name is not None else column.feature_name}' since tokenization was "
                                          f"disabled for this feature (via 'run_tokenization = False').")
                continue

            output_file = self._get_vocabulary_path(context, column)
            create_token_vocabulary(column, main_file, session_index_path,
                                    output_file, self.special_tokens, delimiter, None)

    def _dry_run(self, context: Context) -> None:
        pass

    def dry_run_available(self, context: Context) -> bool:
        for column in self.columns:
            if not column.run_tokenization:
                continue
            else:
                path = self._get_vocabulary_path(context, column)
                if not os.path.exists(path):
                    return False

        return True

    def _get_vocabulary_path(self, context: Context, column: MetaInformation) -> Path:
        output_dir = context.get(OUTPUT_DIR_KEY)
        prefix = format_prefix(context.get(PREFIXES_KEY) if self.prefixes is None else self.prefixes)
        filename = column.column_name if column.column_name is not None else column.feature_name
        if column.get_config("delimiter") is not None:
            filename += "-splitted"

        return output_dir / f"{prefix}.vocabulary.{filename}.txt"


class UseExistingSplit(PreprocessingAction):
    """
    Enables the usage of pre-split datasets from outside of the ASME preprocessing pipeline.
    
    Required context parameters:
    - `SPLIT_BASE_DIRECTORY_PATH` - The path to the existing split. 
    - `PREFIXES_KEY` - The prefixes to use when naming files in the split.
    
    Sets context parameters:
        None
    """
    def __init__(self, split_names: List[str], per_split_actions: List[PreprocessingAction] = None):
        """
        :param split_names: Names of splits to process (e.g. ["train", "validation", "test"].
        :param per_split_actions: A list of action that should be applied for each split.
        """
        if split_names is None or len(split_names) == 0:
            raise Exception(f"At least one name for a split must be supplied.")
        self.split_names = split_names
        self.per_split_actions = [] if per_split_actions is None else per_split_actions

    def name(self) -> str:
        return f"Use existing split."

    def _run(self, context: Context) -> None:

        split_base_directory = context.get(SPLIT_BASE_DIRECTORY_PATH)

        for split_name in self.split_names:
            self._process_split(context, split_base_directory, split_name)

    def _process_split(self,
                       context: Context,
                       split_base_directory: Path,
                       split_name: str):

        # Modify context such that the operations occur in the new output directory using the split file
        cloned = self._prepare_context_for_per_split_actions(context, split_base_directory, split_name)

        # Apply the necessary preprocessing, i.e. session index generation
        for action in self.per_split_actions:
            action._run(cloned)

    @staticmethod
    def _prepare_context_for_per_split_actions(context: Context, split_base_directory: Path, split_name: str) -> Context:
        cloned = copy.deepcopy(context)
        cloned.set(OUTPUT_DIR_KEY, split_base_directory, overwrite=True)

        split_prefix_list = cloned.get(PREFIXES_KEY) + [split_name]
        split_prefix = format_prefix(split_prefix_list)
        cloned.set(MAIN_FILE_KEY, split_base_directory / f"{split_prefix}.csv", overwrite=True)
        cloned.set(PREFIXES_KEY, split_prefix_list, overwrite=True)

        return cloned

    def _dry_run(self, context: Context) -> None:
        pass

    def dry_run_available(self, context: Context) -> bool:
        split_base_directory = context.get(SPLIT_BASE_DIRECTORY_PATH)
        for split_name in self.split_names:
            cloned = self._prepare_context_for_per_split_actions(context, split_base_directory, split_name)
            for action in self.per_split_actions:
                if not action.dry_run_available(cloned):
                    return False

        return True


class CreateRatioSplit(PreprocessingAction):
    """
    Creates a ratio-split using the provided train/validation/test percentages. Each per-split action is executed for
    the train/validation/test split separately while each complete-split action is executed once for the entire ratio-
    split.
    
    Required context parameters:
    - `MAIN_FILE_KEY` - The path to the main CSV file.
    - `OUTPUT_DIR_KEY` - The directory where the ratio-split should be created.
    - `PREFIXES_KEY` - The prefixes to use when naming files of the split.
    - `DELIMITER_KEY` - The delimiter that is used to separate columns in the main CSV file.
    - `SESSION_INDEX_KEY` - The path to the session index to consider to create the ratio split.
    
    Sets context parameters:
    - `RATIO_SPLIT_PATH_CONTEXT_KEY` - The base path to the ratio-split directory.
    
    """
    def __init__(self, train_percentage: float, validation_percentage: float, test_percentage: float,
                 per_split_actions: List[PreprocessingAction] = None,
                 complete_split_actions: List[PreprocessingAction] = None,
                 update_paths: bool = True):
        """
        :param train_percentage: Percentage of the data to use for the training split.
        :param validation_percentage: Percentage of the data to use for the validation split.
        :param test_percentage: Percentage of the data to use for the test split.
        :param per_split_actions: A list of actions that should be executed for each split.
        :param complete_split_actions: A list of actions that should be executed once for the entire split.
        :param update_paths: If this is set to true, the context passed to the complete split actions will be modified,
                             such that all paths (e.g. session index path, main file path, etc.) point to the train
                             split instead of the actual ones for the whole dataset.
        """
        self.test_percentage = test_percentage
        self.validation_percentage = validation_percentage
        self.train_percentage = train_percentage
        self.per_split_actions = [] if per_split_actions is None else per_split_actions
        self.complete_split_actions = [] if complete_split_actions is None else complete_split_actions
        self.update_paths = update_paths

    def name(self) -> str:
        return f"Creating ratio split ({self.train_percentage}/{self.validation_percentage}/{self.test_percentage})"

    def _run(self, context: Context) -> None:
        main_file = context.get(MAIN_FILE_KEY)
        delimiter = context.get(DELIMITER_KEY)
        header = delimiter.join(read_csv_header(main_file, delimiter))
        output_dir = self._get_complete_split_output_dir(context)
        session_index_path = context.get(SESSION_INDEX_KEY)
        session_index = CsvDatasetIndex(session_index_path)
        reader = CsvDatasetReader(main_file, session_index)

        # Create output dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the path to the split in the context for the datamodule to use later
        context.set(RATIO_SPLIT_PATH_CONTEXT_KEY, output_dir)

        # Split the dataset
        split_indices = self._generate_split_indices(session_index, {
            SplitNames.train: self.train_percentage,
            SplitNames.validation: self.validation_percentage,
            SplitNames.test: self.test_percentage
        })

        # Write individual CSVs and execute inner actions for each split
        for name, indices in split_indices.items():
            self._process_split(context, name.name, header, reader, indices)

        self._perform_complete_split_actions(context, output_dir)

    def _dry_run(self, context: Context) -> None:
        output_dir = self._get_complete_split_output_dir(context)
        context.set(RATIO_SPLIT_PATH_CONTEXT_KEY, output_dir)

    def _get_complete_split_output_dir(self, context: Context) -> Path:
        return context.get(OUTPUT_DIR_KEY) / \
                     f"ratio_split-{self.train_percentage}_{self.validation_percentage}_{self.test_percentage}"

    def _get_split_main_file_path(self, context: Context, name: str) -> Path:
        prefix = format_prefix(context.get(PREFIXES_KEY) + [name])
        output_dir = self._get_complete_split_output_dir(context)
        return output_dir / f"{prefix}.csv"

    def dry_run_available(self, context: Context):
        for split in SplitNames:
            # Check whether the split main file exists
            if not os.path.exists(self._get_split_main_file_path(context, split.name)):
                return False

            # Check whether all per split actions are available for dry run
            cloned = self._prepare_context_for_per_split_actions(context, split.name)
            for action in self.per_split_actions:
                if not action.dry_run_available(cloned):
                    return False

        # Check whether all complete split action are available for dry run
        cloned = self._prepare_context_for_complete_split_actions(context)
        for action in self.complete_split_actions:
            if not action.dry_run_available(cloned):
                return False

        # Everything can dry run
        return True

    def _prepare_context_for_per_split_actions(self, context: Context, name: str) -> Context:
        output_dir = self._get_complete_split_output_dir(context)
        output_file = self._get_split_main_file_path(context, name)

        cloned = copy.deepcopy(context)
        cloned.set(OUTPUT_DIR_KEY, output_dir, overwrite=True)
        cloned.set(MAIN_FILE_KEY, output_file, overwrite=True)
        current_prefixes = context.get(PREFIXES_KEY)
        cloned.set(PREFIXES_KEY, current_prefixes + [name], overwrite=True)

        return cloned

    def _prepare_context_for_complete_split_actions(self, context: Context) -> Context:
        cloned = copy.deepcopy(context)
        output_dir = self._get_complete_split_output_dir(context)
        cloned.set(OUTPUT_DIR_KEY, output_dir, overwrite=True)
        # If enabled, change paths to point to the train split, e.g. for creating the vocabulary or popularities
        if self.update_paths:
            current_prefixes = context.get(PREFIXES_KEY)
            train_prefix = current_prefixes + [SplitNames.train.name]
            cloned.set(PREFIXES_KEY, train_prefix, overwrite=True)
            cloned.set(MAIN_FILE_KEY, output_dir / f"{format_prefix(train_prefix)}.csv", overwrite=True)
            cloned.set(SESSION_INDEX_KEY, output_dir / f"{format_prefix(train_prefix)}.session.idx",
                       overwrite=True)

        return cloned

    @staticmethod
    def _generate_split_indices(session_index: CsvDatasetIndex, split_ratios: Dict[SplitNames, float]):
        length = len(session_index)
        indices = list(range(length))
        random.shuffle(indices)

        splits = dict()
        remainder = indices
        for split_name, ratio in split_ratios.items():
            remainder_length = len(remainder)
            num_samples_in_split = int(math.ceil(ratio * length))

            # take only what is left for the last split to avoid errors
            num_samples_in_split = min(num_samples_in_split, remainder_length)

            samples = remainder[:num_samples_in_split]
            remainder = remainder[num_samples_in_split:]

            splits[split_name] = samples

        return TrainValidationTestSplitIndices(train_indices=splits[SplitNames.train],
                                               validation_indices=splits[SplitNames.validation],
                                               test_indices=splits[SplitNames.test])

    def _process_split(self, context: Context, name: str, header: str, reader: CsvDatasetReader,
                       indices: List[int]):
        output_file = self._get_split_main_file_path(context, name)

        # Write the CSV file for the current split
        self._write_split(output_file, header, reader, indices)

        # Modify context such that the operations occur in the new output directory using the split file
        cloned = self._prepare_context_for_per_split_actions(context, name)

        # Apply the necessary preprocessing, i.e. session index generation
        for action in self.per_split_actions:
            action._run(cloned)

    def _perform_complete_split_actions(self, context: Context, split_output_dir: Path):
        # Modify context such that the operations occur in the new output directory
        cloned = self._prepare_context_for_complete_split_actions(context)

        for action in self.complete_split_actions:
            action._run(cloned)

    @staticmethod
    def _write_split(output_file: Path, header: str, reader: CsvDatasetReader, indices: List[int]):
        with open(output_file, "w") as f:
            f.write(header + "\n")
            for index in indices:
                sample = reader.get_sequence(index)
                f.write(sample.strip() + "\n")

    @staticmethod
    def _get_header(input_file: Path) -> str:
        with open(input_file, "r") as f:
            return f.readline().strip()


class CreatePopularity(PreprocessingAction):
    """
    Creates item popularities for each of the provided columns.
   
    Required context parameters:
    - `MAIN_FILE_KEY` - The path to the main CSV file.
    - `OUTPUT_DIR_KEY` - The directory where the popularity files should be saved.
    - `PREFIXES_KEY` - The prefixes to use when naming the popularity files.
    - `DELIMITER_KEY` - The delimiter that is used to separate columns in the main CSV file.
    - `SESSION_INDEX_KEY` - The path to the session index to consider to create the popularities.
    
    Sets context parameters:
        None
    """
    def __init__(self, columns: List[MetaInformation], prefixes: List[str] = None, special_tokens: Dict[str, str] = None):
        """
        :param columns: A list of all columns for which a popularity should be generated. Note that a column is skipped,
                        if its `run_tokenization` flag is set to false.
        :param special_tokens: A mapping of special tokens to pass to the tokenizer when counting the occurrence of each
                               token. Per default the following special tokens are supplied:
                               { 'pad_token': '<PAD>',
                                 'unk_token': '<UNK>',
                                 'mask_token': '<MASK>' }
        :param prefixes: Allows to overwrite the prefixes used for naming the popularity files. If None is passed, the
                         prefixes are determined from the context.
        """
        self.columns = columns
        self.prefixes = prefixes
        self.special_tokens = {
            "pad_token": "<PAD>",
            "mask_token": "<MASK>",
            "unk_token": "<UNK>",
        } if special_tokens is None else special_tokens

    def name(self) -> str:
        return f"Creating popularities for: {self.columns}"

    def _run(self, context: Context) -> None:
        main_file = context.get(MAIN_FILE_KEY)
        output_dir = context.get(OUTPUT_DIR_KEY)
        delimiter = context.get(DELIMITER_KEY)
        prefixes = context.get(PREFIXES_KEY) if self.prefixes is None else self.prefixes
        prefix = format_prefix(prefixes)
        header = create_indexed_header(read_csv_header(main_file, delimiter))
        session_index_path = context.get(SESSION_INDEX_KEY)
        session_index = CsvDatasetIndex(session_index_path)
        reader = CsvDatasetReader(main_file, session_index)
        for column in self.columns:
            filename = column.column_name if column.column_name is not None else column.feature_name

            if not column.run_tokenization:
                get_root_logger().warning(f"Skipping popularity generation for '{filename}' since tokenization was "
                                          f"disabled for this feature (via 'run_tokenization = False').")
                continue
            session_parser = ItemSessionParser(header, [column], delimiter=delimiter)
            plain_dataset = PlainSequenceDataset(reader, session_parser)
            dataset = ItemSequenceDataset(plain_dataset)
            sub_delimiter = column.get_config("delimiter")

            if column.get_config("delimiter") is not None:
                filename += "-splitted"

            # Read the vocabulary for this column
            vocabulary_path = output_dir / f"{prefix}.vocabulary.{filename}.txt"
            with open(vocabulary_path, "r") as f:
                vocabulary = CSVVocabularyReaderWriter().read(f)

            # Get occurrence count of every token
            counts = self._count_items(dataset, vocabulary, column, sub_delimiter)
            # Compute popularity
            total_count = sum(counts.values())
            popularities = [count / total_count for count in counts.values()]
            # Save them to the correct file
            output_file = output_dir / f"{prefix}.popularity.{filename}.txt"
            self._write_popularity(popularities, output_file)

    def _dry_run(self, context: Context) -> None:
        pass

    def dry_run_available(self, context: Context) -> bool:
        for column in self.columns:
            if not column.run_tokenization:
                continue
            if not os.path.exists(self._get_popularity_path(context, column)):
                return False

        return True

    def _get_popularity_path(self, context: Context, column: MetaInformation) -> Path:
        output_dir = context.get(OUTPUT_DIR_KEY)
        filename = column.column_name if column.column_name is not None else column.feature_name
        if column.get_config("delimiter") is not None:
            filename += "-splitted"
        prefixes = context.get(PREFIXES_KEY) if self.prefixes is None else self.prefixes
        prefix = format_prefix(prefixes)
        return output_dir / f"{prefix}.popularity.{filename}.txt"

    def _count_items(self, dataset: ItemSequenceDataset, vocabulary: Vocabulary, column: MetaInformation, sub_delimiter: str = None) -> Dict[int, int]:

        tokenizer = Tokenizer(vocabulary, **self.special_tokens)
        counts = defaultdict(int)

        for session_idx in range(len(dataset)):
            session = dataset[session_idx]
            items = session[column.feature_name]
            if sub_delimiter is not None:
                items = [label for entry in items for label in entry.split(sub_delimiter)]
            converted_tokens = tokenizer.convert_tokens_to_ids(items)
            for token in converted_tokens:
                counts[token] += 1

        # Include special tokens in popularity
        for special_token_id in tokenizer.get_special_token_ids():
            if special_token_id not in counts:
                counts[special_token_id] = 0

        return counts

    @staticmethod
    def _write_popularity(popularities: List[float], output_file: Path):
        with open(output_file, 'w') as f:
            for popularity in popularities:
                f.write(f"{popularity}\n")
