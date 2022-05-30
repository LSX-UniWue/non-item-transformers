import os
from pathlib import Path
from typing import List

from tqdm import tqdm

from asme.core.init.context import Context
from asme.core.tokenization.vocabulary import VocabularyBuilder, CSVVocabularyReaderWriter
from asme.core.utils.logging import get_root_logger
from asme.data.datamodule.preprocessing.action import PreprocessingAction, MAIN_FILE_KEY, SESSION_INDEX_KEY, \
    DELIMITER_KEY, OUTPUT_DIR_KEY, PREFIXES_KEY
from asme.data.datamodule.preprocessing.util import format_prefix, create_session_data_set
from asme.data.datasets.sequence import MetaInformation


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
        columns = list(map(lambda col: col.column_name if col.column_name is not None else col.feature_name, self.columns))
        return f"Creating Vocabulary for columns: {columns}."

    def _run(self, context: Context) -> None:
        main_file = context.get(MAIN_FILE_KEY)
        session_index_path = context.get(SESSION_INDEX_KEY)
        delimiter = context.get(DELIMITER_KEY)
        for column in self.columns:
            if not column.run_tokenization:
                get_root_logger().warning(
                    f"Skipping vocabulary generation for '{column.column_name if column.column_name is not None else column.feature_name}' since tokenization was "
                    f"disabled for this feature (via 'run_tokenization = False').")
                continue

            output_file = self._get_vocabulary_path(context, column)
            self._create_token_vocabulary(column, main_file, session_index_path,
                                          output_file, delimiter)

    def _create_token_vocabulary(self, column: MetaInformation,
                                 data_file_path: Path,
                                 session_index_path: Path,
                                 vocabulary_output_file_path: Path,
                                 delimiter: str):

        vocab_builder = VocabularyBuilder()
        for token in self.special_tokens:
            vocab_builder.add_token(token)

        sub_delimiter = column.get_config("delimiter")

        data_set = create_session_data_set(column=column,
                                           data_file_path=data_file_path,
                                           index_file_path=session_index_path,
                                           delimiter=delimiter)

        feature_name = column.feature_name if column.feature_name else column.column_name

        for idx in tqdm(range(len(data_set)), desc=f"Tokenizing feature '{feature_name}' from: {data_file_path}"):
            session = data_set[idx]
            session_tokens = session[feature_name]

            for token in session_tokens:
                if isinstance(token, list):
                    for word in token:
                        vocab_builder.add_token(word)
                else:
                    vocab_builder.add_token(token)

        vocabulary = vocab_builder.build()

        if not os.path.exists(vocabulary_output_file_path.parent):
            vocabulary_output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with vocabulary_output_file_path.open("w") as file:
            CSVVocabularyReaderWriter().write(vocabulary, file)

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
            filename += "-split"

        return output_dir / f"{prefix}.vocabulary.{filename}.txt"