import lzma
import os
import tarfile
import zipfile
from abc import abstractmethod
from pathlib import Path

from asme.core.utils.logging import get_logger


class Unpacker:
    """
    Base class for all dataset unpackers.
    """

    def __init__(self, target_directory: Path):
        """
        :param target_directory: The directory where the extracted files wil be saved.
        """
        self._target_directory = target_directory

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def apply(self, location: Path, force_unpack=False) -> Path:
        """
        Unpack the dataset located a location.

        :param location: The location of the packed dataset. This might be a path to a file or directory depending on
        the type of packaging used.
        :param force_unpack: Indicated whether an existing unpacked dataset at the target location should be overwritten.
        """
        pass

    def __call__(self, location: Path, force_unpack=False) -> Path:
        return self.apply(location, force_unpack=force_unpack)


class Unzipper(Unpacker):
    """
    Unpacker for zipped datasets.
    """

    def __init__(self, target_directory: Path):
        """
        :param target_directory: The directory where the extracted files wil be saved.
        """
        super(Unzipper, self).__init__(target_directory)

    def name(self) -> str:
        return "Dataset Unzipper"

    def apply(self, location: Path, force_unpack=False) -> Path:
        with zipfile.ZipFile(location) as zip_file:
            if force_unpack:
                zip_file.extractall(self._target_directory)
            else:
                for zf in zip_file.infolist():
                    extraction_path = self._target_directory / zf.filename
                    if not extraction_path.exists():
                        zip_file.extract(zf, self._target_directory)

        return self._target_directory


class TarXzUnpacker(Unpacker):
    """
    Unpacker for datasets compressed as *.tar.xz (i.e using LZMA)
    """

    def __init__(self, target_directory: Path):
        super(TarXzUnpacker, self).__init__(target_directory)

    def name(self) -> str:
        return "Dataset XZ Unpacker"

    def apply(self, location: Path, force_unpack=False) -> Path:
        with tarfile.open(location) as tar_file:
            if force_unpack or not self._target_directory.exists() or len(os.listdir(self._target_directory)) == 0:
                tar_file.extractall(self._target_directory)
            else:
                get_logger("XZ Unpacker").warning(f"The dataset target directory is non-empty and force-regeneration = False. Not doing anything!")
        return self._target_directory
