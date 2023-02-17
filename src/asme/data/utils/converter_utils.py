import datetime
import functools
from typing import Callable, Any, List, Dict
import json


def build_converter(feature_type: str,
                    configs: Dict[str, Any]
                    ) -> Callable[[str], Any]:
    if feature_type == 'int':
        return int

    if feature_type == 'float':
        return float

    if feature_type == 'str':
        return _identity

    if feature_type == 'bool':
        return _parse_boolean

    if feature_type == 'timestamp':
        return functools.partial(_parse_timestamp, date_format=configs.get('format'))

    if feature_type == 'list':
        element_type = configs.get("element_type")
        delimiter = configs.get('delimiter')
        converter = build_converter(element_type, configs)
        return functools.partial(_parse_list, delimiter=delimiter, converter=converter)

    if feature_type == 'pd.array':
        element_type = configs.get("element_type")
        converter = build_converter(element_type, configs)
        return functools.partial(_parse_pd_array, converter=converter)

    raise KeyError(f'{feature_type} not supported. Currently bool, str, float, timestamp, int, pd.array and list are supported. '
                   f'See documentation for more details')


def _parse_boolean(text: str
                   ) -> bool:
    return text == 'True'


def _parse_timestamp(text: str,
                     date_format: str
                     ) -> datetime.datetime:
    return datetime.datetime.strptime(text, date_format)


def _parse_list(text: str,
                converter: Callable[[str],Any],
                delimiter: str) -> List[str]:
    return list(map(converter, text.split(sep=delimiter)))


def _parse_pd_array(text: str,
                converter: Callable[[str], Any]) -> List[str]:
    text = json.loads(text)
    #text = text.strip('][').replace('"', '').split(',')
    # text = ast.literal_eval(text)
    return list(map(converter, text))


def _identity(text: str
              ) -> str:
    return text
