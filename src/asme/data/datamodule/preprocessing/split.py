import copy
import math
import os
import random
from pathlib import Path
from typing import List, Dict

from asme.core.init.context import Context
from asme.core.init.templating.datasources.datasources import DatasetSplit
from asme.data import LOO_SPLIT_PATH_CONTEXT_KEY, CURRENT_SPLIT_PATH_CONTEXT_KEY, RATIO_SPLIT_PATH_CONTEXT_KEY, \
    LPO_SPLIT_PATH_CONTEXT_KEY
from asme.data.base.reader import CsvDatasetIndex, CsvDatasetReader
from asme.data.datamodule.extractors import FixedOffsetPositionExtractor, PercentageBasedPositionExtractor
from asme.data.datamodule.preprocessing.action import PreprocessingAction, MAIN_FILE_KEY, SESSION_INDEX_KEY, \
    DELIMITER_KEY, PREFIXES_KEY, OUTPUT_DIR_KEY
from asme.data.datamodule.preprocessing.util import format_prefix
from asme.data.datamodule.util import SplitNames, TrainValidationTestSplitIndices
from asme.data.datasets.index_builder import SequencePositionIndexBuilder
from asme.data.datasets.sequence import ItemSessionParser, ItemSequenceDataset, PlainSequenceDataset, MetaInformation
from asme.data.utils.csv import create_indexed_header, read_csv_header


class CreateLeavePercentageOutSplit(PreprocessingAction):
    def __init__(self, column: MetaInformation, train_percentage=0.8, validation_percentage=0.1,
                 test_percentage=0.1, min_train_length=2, min_validation_lenght=1, min_test_length=1,
                 inner_actions: List[PreprocessingAction] = None):

        self._column = column
        self._inner_actions = [] if inner_actions is None else inner_actions

        self._train_percentage = train_percentage
        self._validation_percentage = validation_percentage
        self._test_percentage = test_percentage

        self._min_test_length = min_test_length
        self._min_validation_length = min_validation_lenght
        self._min_train_length = min_train_length

        # Need to use math.isclose since the check might fail due to numerical precision otherwise.
        if not math.isclose(train_percentage + test_percentage + validation_percentage, 1.):
            raise ValueError(f"Fractions for training, validation and test do not sum to 1"
                             f" (got {train_percentage}/{validation_percentage}/{test_percentage}).")

    def name(self) -> str:
        return "Creating leave-percentage-out split"

    def _run(self, context: Context) -> None:
        output_dir = context.get(OUTPUT_DIR_KEY)
        main_file = context.get(MAIN_FILE_KEY)
        session_index_path = context.get(SESSION_INDEX_KEY)
        delimiter = context.get(DELIMITER_KEY)

        output_dir.mkdir(parents=True, exist_ok=True)
        # Save the path to the split in the context for the datamodule to use later
        context.set(LPO_SPLIT_PATH_CONTEXT_KEY, output_dir)

        session_index = CsvDatasetIndex(session_index_path)
        reader = CsvDatasetReader(main_file, session_index)
        parser = ItemSessionParser(
            create_indexed_header(read_csv_header(main_file, delimiter)),
            [self._column],
            delimiter=delimiter
        )
        dataset = ItemSequenceDataset(PlainSequenceDataset(reader, parser))
        for name, offset in zip([SplitNames.train, SplitNames.validation, SplitNames.test],
                                [self._train_percentage, self._validation_percentage, self._test_percentage]):
            prefix = format_prefix(context.get(PREFIXES_KEY) + [name.name])
            output_file = output_dir / f"{prefix}.loo.idx"
            builder = SequencePositionIndexBuilder(target_positions_extractor=PercentageBasedPositionExtractor(
                self._train_percentage, self._validation_percentage, self._test_percentage,
                name,
                self._min_train_length, self._min_validation_length, self._min_test_length
            ))
            builder.build(dataset, output_file)

        cloned = self._prepare_context_for_complete_split_actions(context)
        for action in self._inner_actions:
            action(cloned)

    def _dry_run(self, context: Context) -> None:
        context.set(LPO_SPLIT_PATH_CONTEXT_KEY, context.get(OUTPUT_DIR_KEY))

    def dry_run_available(self, context: Context) -> bool:
        cloned = self._prepare_context_for_complete_split_actions(context)
        for action in self._inner_actions:
            if not action.dry_run_available(cloned):
                return False

        return True

    def _prepare_context_for_complete_split_actions(self, context: Context) -> Context:
        return copy.deepcopy(context)


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

    def __init__(self, column: MetaInformation, training_target_offset=2, validation_target_offset=1,
                 test_target_offset=0,
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
        return "Creating leave-one-out split"

    def _run(self, context: Context) -> None:
        output_dir = context.get(OUTPUT_DIR_KEY)
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
        context.set(LOO_SPLIT_PATH_CONTEXT_KEY, context.get(OUTPUT_DIR_KEY))

    def dry_run_available(self, context: Context) -> bool:
        for action in self.inner_actions:
            if not action.dry_run_available(context):
                return False

        return True

    def _prepare_context_for_complete_split_actions(self, context: Context):
        return copy.deepcopy(context)


class UseExistingSplit(PreprocessingAction):
    """
    Enables the usage of pre-split datasets from outside of the ASME preprocessing pipeline.

    Required context parameters:
    - `SPLIT_BASE_DIRECTORY_PATH` - The path to the existing split.
    - `PREFIXES_KEY` - The prefixes to use when naming files in the split.

    Sets context parameters:
    - `RATIO_SPLIT_PATH_CONTEXT_KEY` or `LOO_SPLIT_PATH_CONTEXT_KEY` depending on which split type was specified.
    """

    def __init__(self,
                 split_names: List[str],
                 split_type: DatasetSplit,
                 complete_split_actions: List[PreprocessingAction] = None,
                 per_split_actions: List[PreprocessingAction] = None,
                 update_paths: bool = True):
        """
        :param split_names: Names of splits to process (e.g. ["train", "validation", "test"].
        :param per_split_actions: A list of action that should be applied for each split.
        """
        if split_names is None or len(split_names) == 0:
            raise Exception(f"At least one name for a split must be supplied.")
        self.split_names = split_names
        self.split_type = split_type
        self.complete_split_actions = [] if complete_split_actions is None else complete_split_actions
        self.per_split_actions = [] if per_split_actions is None else per_split_actions
        self.update_paths = update_paths

    def name(self) -> str:
        return f"Use existing split."

    def _run(self, context: Context) -> None:

        split_base_directory = context.get(CURRENT_SPLIT_PATH_CONTEXT_KEY)

        if self.split_type == DatasetSplit.RATIO_SPLIT:
            context.set(RATIO_SPLIT_PATH_CONTEXT_KEY, split_base_directory)
        else:
            context.set(LOO_SPLIT_PATH_CONTEXT_KEY, split_base_directory)

        for split_name in self.split_names:
            self._process_split(context, split_base_directory, split_name)

        self._process_complete_split(context, split_base_directory)

    def _process_complete_split(self,
                                context: Context,
                                split_base_path_directory: Path):

        cloned = self._prepare_context_for_complete_split_actions(context, split_base_path_directory)

        # Apply the necessary preprocessing, i.e. session index generation
        for action in self.complete_split_actions:
            action(cloned)

    def _process_split(self,
                       context: Context,
                       split_base_directory: Path,
                       split_name: str):

        # Modify context such that the operations occur in the new output directory using the split file
        cloned = self._prepare_context_for_per_split_actions(context, split_base_directory, split_name)

        # Apply the necessary preprocessing, i.e. session index generation
        for action in self.per_split_actions:
            action(cloned)

    @staticmethod
    def _prepare_context_for_per_split_actions(context: Context, split_base_directory: Path,
                                               split_name: str) -> Context:
        cloned = copy.deepcopy(context)
        cloned.set(OUTPUT_DIR_KEY, split_base_directory, overwrite=True)

        split_prefix_list = cloned.get(PREFIXES_KEY) + [split_name]
        split_prefix = format_prefix(split_prefix_list)
        cloned.set(MAIN_FILE_KEY, split_base_directory / f"{split_prefix}.csv", overwrite=True)
        cloned.set(PREFIXES_KEY, split_prefix_list, overwrite=True)

        return cloned

    def _prepare_context_for_complete_split_actions(self, context: Context, split_base_directory: Path) -> Context:
        cloned = copy.deepcopy(context)
        cloned.set(OUTPUT_DIR_KEY, split_base_directory, overwrite=True)
        # If enabled, change paths to point to the train split, e.g. for creating the vocabulary or popularities
        if self.update_paths:
            current_prefixes = context.get(PREFIXES_KEY)
            train_prefix = current_prefixes + [SplitNames.train.name]
            cloned.set(PREFIXES_KEY, train_prefix, overwrite=True)
            cloned.set(MAIN_FILE_KEY, split_base_directory / f"{format_prefix(train_prefix)}.csv", overwrite=True)
            cloned.set(SESSION_INDEX_KEY, split_base_directory / f"{format_prefix(train_prefix)}.session.idx",
                       overwrite=True)

        return cloned

    def _dry_run(self, context: Context) -> None:
        split_base_directory = context.get(CURRENT_SPLIT_PATH_CONTEXT_KEY)
        if self.split_type == DatasetSplit.RATIO_SPLIT:
            context.set(RATIO_SPLIT_PATH_CONTEXT_KEY, split_base_directory)
        else:
            context.set(LOO_SPLIT_PATH_CONTEXT_KEY, split_base_directory)

    def dry_run_available(self, context: Context) -> bool:
        split_base_directory = context.get(CURRENT_SPLIT_PATH_CONTEXT_KEY)
        for split_name in self.split_names:
            cloned = self._prepare_context_for_per_split_actions(context, split_base_directory, split_name)
            for action in self.per_split_actions:
                if not action.dry_run_available(cloned):
                    return False

        cloned = self._prepare_context_for_complete_split_actions(context, split_base_directory)
        for action in self.complete_split_actions:
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
        output_dir = context.get(OUTPUT_DIR_KEY)
        main_file = context.get(MAIN_FILE_KEY)
        delimiter = context.get(DELIMITER_KEY)
        header = delimiter.join(read_csv_header(main_file, delimiter))
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
        output_dir = context.get(OUTPUT_DIR_KEY)
        context.set(RATIO_SPLIT_PATH_CONTEXT_KEY, output_dir)

    def _get_split_main_file_path(self, context: Context, name: str) -> Path:
        prefix = format_prefix(context.get(PREFIXES_KEY) + [name])
        output_dir = context.get(OUTPUT_DIR_KEY)
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
        output_file = self._get_split_main_file_path(context, name)

        cloned = copy.deepcopy(context)
        cloned.set(MAIN_FILE_KEY, output_file, overwrite=True)
        current_prefixes = context.get(PREFIXES_KEY)
        cloned.set(PREFIXES_KEY, current_prefixes + [name], overwrite=True)

        return cloned

    def _prepare_context_for_complete_split_actions(self, context: Context) -> Context:
        cloned = copy.deepcopy(context)
        output_dir = context.get(OUTPUT_DIR_KEY)
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
            action(cloned)

    def _perform_complete_split_actions(self, context: Context, split_output_dir: Path):
        # Modify context such that the operations occur in the new output directory
        cloned = self._prepare_context_for_complete_split_actions(context)

        for action in self.complete_split_actions:
            action(cloned)

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
