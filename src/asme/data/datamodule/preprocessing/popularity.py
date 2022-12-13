import os
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

from asme.core.init.context import Context
from asme.core.tokenization.tokenizer import Tokenizer
from asme.core.tokenization.vocabulary import CSVVocabularyReaderWriter, Vocabulary
from asme.core.utils.logging import get_root_logger
from asme.data.base.reader import CsvDatasetIndex, CsvDatasetReader
from asme.data.datamodule.preprocessing.action import PreprocessingAction, MAIN_FILE_KEY, OUTPUT_DIR_KEY, \
    DELIMITER_KEY, PREFIXES_KEY, SESSION_INDEX_KEY
from asme.data.datamodule.preprocessing.util import format_prefix
from asme.data.datasets.sequence import ItemSessionParser, PlainSequenceDataset, ItemSequenceDataset, MetaInformation
from asme.data.utils.csv import create_indexed_header, read_csv_header


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

    def __init__(self, columns: List[MetaInformation], prefixes: List[str] = None,
                 special_tokens: Dict[str, str] = None):
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
                filename += "-split"

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
            filename += "-split"
        prefixes = context.get(PREFIXES_KEY) if self.prefixes is None else self.prefixes
        prefix = format_prefix(prefixes)
        return output_dir / f"{prefix}.popularity.{filename}.txt"

    def _count_items(self, dataset: ItemSequenceDataset, vocabulary: Vocabulary, column: MetaInformation,
                     sub_delimiter: str = None) -> Dict[int, int]:

        tokenizer = Tokenizer(vocabulary, **self.special_tokens)
        counts = defaultdict(int)

        for session_idx in range(len(dataset)):
            session = dataset[session_idx]
            items = session[column.feature_name]
            flat_items = []
            for item in items:
                if isinstance(item, list):
                    flat_items.extend(item)
                else:
                    flat_items += [item]
            converted_tokens = tokenizer.convert_tokens_to_ids(flat_items)
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