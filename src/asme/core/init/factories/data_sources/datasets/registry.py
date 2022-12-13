from typing import List

from asme.core.init.factories.data_sources.datasets.dataset_factory import DatasetFactory
from asme.core.init.factories.data_sources.datasets.item_session import ItemSessionDatasetFactory
from asme.core.init.factories.data_sources.datasets.placeholder import PlaceholderDatasetFactory
from asme.core.init.factories.data_sources.datasets.sequence_position import SequencePositionDatasetFactory

REGISTERED_DATASETS = {}

def register_dataset_factory(key: str, dataset_factory: DatasetFactory,  overwrite: bool = False):
    if key in REGISTERED_DATASETS and not overwrite:
        raise KeyError(f"A dataset with key '{key}' is already registered and overwrite was set to false.")
    REGISTERED_DATASETS[key] = dataset_factory

def get_dataset_factories() -> List[DatasetFactory]:
    return REGISTERED_DATASETS.values()

register_dataset_factory("placeholder", PlaceholderDatasetFactory())
register_dataset_factory("sequence_position", SequencePositionDatasetFactory())
register_dataset_factory("session", ItemSessionDatasetFactory())
