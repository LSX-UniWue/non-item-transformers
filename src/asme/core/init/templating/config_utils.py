from typing import Dict, Any, List, Optional


def config_entry_exists(config: Dict[str, Any], path: List[str]) -> bool:
    return get_config_value(config, path) is not None


def get_config_value(config: Dict[str, Any], path: List[str]) -> Optional[Any]:
    cursor = config
    for element in path:
        if cursor is None:
            return False

        cursor = cursor.get(element)

    return cursor


def set_config_value(config: Dict[str, Any], path: List[str], value: Any, make_parents: bool = True) -> bool:
    cursor = config

    for element in path[:-1]:
        if not isinstance(cursor, Dict):
            return False

        if element not in cursor:
            if make_parents:
                cursor[element] = dict()
            else:
                return False

        cursor = cursor[element]

    if not isinstance(cursor, dict):
        return False

    cursor[path[-1]] = value

    return True
