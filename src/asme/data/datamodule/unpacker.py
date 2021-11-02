import lzma
import tarfile
import zipfile
from abc import abstractmethod
from pathlib import Path


class Unpacker:
    """
    Base class for all dataset unpackers.
    """

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def apply(self, location: Path) -> Path:
        """
        Unpack the dataset located a location.

        :param location: The location of the packed dataset. This might be a path to a file or directory depending on
        the type of packaging used.
        """
        pass

    def __call__(self, location: Path) -> Path:
        return self.apply(location)


class Unzipper(Unpacker):
    """
    Unpacker for zipped datasets.
    """

    def __init__(self, target_directory: Path):
        """
        :param target_directory: The directory where the extracted files wil be saved.
        """
        self._target_directory = target_directory

    def name(self) -> str:
        return "Dataset Unzipper"

    def apply(self, location: Path) -> Path:
        with zipfile.ZipFile(location) as zip_file:
            zip_file.extractall(self._target_directory)

        return self._target_directory


class TarXzUnpacker(Unpacker):
    """
    Unpacker for datasets compressed as *.tar.xz (i.e using LZMA)
    """

    def __init__(self, target_directory: Path):
        """
        :param target_directory: The directory where the extracted files wil be saved.
        """
        self._target_directory = target_directory

    def name(self) -> str:
        return "Dataset XZ Unpacker"

    def apply(self, location: Path) -> Path:
        with tarfile.open(location) as tar_file:
            tar_file.extractall(self._target_directory)

        return self._target_directory
