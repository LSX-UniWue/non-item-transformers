from pathlib import Path
from typing import List

from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.factories.util import require_config_keys
from asme.core.init.object_factory import ObjectFactory, CanBuildResult
from asme.core.tokenization.vocabulary import CSVVocabularyReaderWriter


class VocabularyFactory(ObjectFactory):
    """
    Builds a vocabulary.
    """
    KEY = "vocabulary"
    REQUIRED_KEYS = ["file"]

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        return require_config_keys(build_context.get_current_config_section(), self.REQUIRED_KEYS)

    def build(self, build_context: BuildContext):
        config = build_context.get_current_config_section()
        delimiter = config.get_or_default("delimiter", "\t")
        vocab_file = config.get_or_raise("file", f"<file> could not be found in vocabulary config section.")

        vocab_reader = CSVVocabularyReaderWriter(delimiter)

        with Path(vocab_file).open("r") as file:
            return vocab_reader.read(file)

    def is_required(self, build_context: BuildContext) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
