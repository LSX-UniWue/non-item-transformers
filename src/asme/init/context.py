from typing import List, Any, Optional, Union


class Context:
    """
    Manages the factory context. The context maps string keys to objects. Hierarchies are expressed by separating the
    section with dots, e.g. `tokenizers.item` or `data_sources.train`.
    """
    def __init__(self):
        self.context = {}

    def _path_to_str(self, path: List[str]) -> str:
        return ".".join(path)

    def _str_to_path(self, path: str) -> List[str]:
        return path.split(".")

    def set(self, path: Union[str, List[str]], value: Any, overwrite: bool = False):
        if isinstance(path, list):
            path = self._path_to_str(path)
        if path in self.context and not overwrite:
            raise Exception(f"{path} already exists in {self.context}")
        self.context[path] = value

    def get(self, path: Union[str, List[str]]) -> Optional[Any]:
        if isinstance(path, list):
            path = self._path_to_str(path)
        if path not in self.context:
            return None
        else:
            return self.context[path]

    def has_path(self,
                 path: Union[str, List[str]]
                 ) -> bool:
        if isinstance(path, list):
            path = self._path_to_str(path)
        return path in self.context

    def as_dict(self):
        return dict(self.context)
