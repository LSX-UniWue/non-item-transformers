from typing import Dict, Any, List, Optional, Union


class Config:
    """
    An object that controls access to the configuration.
    """

    def __init__(self,
                 config: Dict[str, Any],
                 base_path: List[str] = None):
        """
        Initializes the configuration object.
        :param config: a dictionary that contains the configuration values.
        :param base_path: the path pointing from the root of the configuration to this sub configuration.
        """
        if base_path is None:
            base_path = []
        self.config = config
        self.base_path = base_path

    def _get_path(self, path: List[str]) -> Any:
        # make sure that if a string is accidentally supplied it is treated correctly as a path of length = 1
        if isinstance(path, str):
            path = [path]

        current_section = self.config

        for i, key in enumerate(path):
            if not isinstance(current_section, Dict) or key not in current_section:
                return None
            current_section = current_section[key]

        return current_section

    def set(self, path: Union[str, List[str]], value: Any, make_parents: bool = True) -> Optional[Any]:
        if isinstance(path, str):
            path = path.split(".")

        current_section = self.config

        for i, key in enumerate(path[:-1]):
            if not isinstance(current_section, Dict):
                return None
            if key not in current_section and make_parents:
                current_section[key] = dict()
            else:
                return None

            current_section = current_section[key]

        if not isinstance(current_section, dict):
            return None

        current_section[path[-1]] = value

        return value

    def set_if_absent(self, path: Union[str, List[str]], value: Any) -> None:
        """
        Saves the provided path to the value if it does not exist already.

        :param path: a path.
        :param value: The new value for path.
        """

        if self.has_path(path):
            return

        self.set(path, value)

    def get(self,
            path: Union[str, List[str]]
            ) -> Optional[Any]:
        """
        Gets the value associated with a path. If the path does not exist within the configuration, None is returned.

        :param path: a path.
        :return: a value, or None if the path does not exist.
        """
        return self._get_path(path)

    def pop(self, path: Union[str, List[str]]) -> Optional[Any]:
        """
        Gets the value associated with a path and deletes it afterwards. If the path is not found, None is returned.

        :param path: a path.
        :return; a value, or None if the path does not exist
        """
        value = self.get(path)
        self.remove(path)
        return value

    def pop_or_default(self, path: Union[str, List[str]], default: Any) -> Any:
        """
        Gets the value associated with a path and removes it or returns the default if it does not exist

        :param path: a path.
        :param default: default value to use if the path could not be found.

        :return: a value.
        """
        value = self.pop(path)

        if value is None:
            return default

        return value


    def has_path(self, path: Union[str, List[str]]) -> bool:
        """
        Checks whether a path exists in the configuration.

        :param path: a path.
        :return: :code true if the path exists, :code false otherwise.
        """
        return self.get(path) is not None

    def get_or_raise(self, path: Union[str, List[str]], exception_msg: str = None) -> Any:
        """
        Gets the value associated with a key or raises an exception if the key can not be found.

        :param path: a path.
        :param exception_msg: a message for the exception.
        :return: a value.
        """
        value = self._get_path(path)

        if value is None:
            raise Exception(exception_msg)

        return value

    def get_or_default(self, path: Union[str, List[str]], default: Any) -> Any:
        """
        Gets the value associated with a path or the default value if the key can not be found.

        :param path: a path.
        :param default: a default value.
        :return: a value.
        """
        value = self._get_path(path)

        if value is None:
            return default

        return value

    def remove(self, path: Union[str, List[str]]):
        """
        Removes the value associated with a path

        :param path: a path.
        """
        if self.has_path(path):
            if isinstance(path, str):
                path = path.split(".")

            current_section = self.config

            for i, key in enumerate(path[:-1]):
                if not isinstance(current_section, Dict) or key not in current_section:
                    return
                else:
                    current_section = current_section[key]

            current_section.pop(path[-1])

    def get_config(self, path: List[str]) -> 'Config':
        if not isinstance(path, list):
            raise Exception(f"{path} must be a list!")

        base_path = list(self.base_path)
        base_path.extend(path)
        return Config(self.get(path), base_path)

    def is_root(self) -> bool:
        return len(self.base_path) < 0

    def get_keys(self) -> List[str]:
        """
        Returns a list with all keys at the root of this config instance.

        :return: a list with keys.
        """
        return [key for key in self.config.keys()]
