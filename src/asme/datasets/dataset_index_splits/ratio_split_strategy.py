import math
import numpy as np

from typing import Dict

from asme.data.base.reader import CsvDatasetIndex
from asme.datasets.data_structures.dataset_metadata import DatasetMetadata
from asme.datasets.data_structures.split_strategy import SplitStrategy
from asme.datasets.data_structures.split_names import SplitNames
from asme.datasets.data_structures.train_validation_test_splits_indices import TrainValidationTestSplitIndices


class RatioSplitStrategy(SplitStrategy):
    def __init__(self, train_ratio: float, test_ratio: float, validation_ratio: float, seed: int):
        self.split_ratios: Dict[SplitNames, float] = {SplitNames.train: train_ratio,
                                                      SplitNames.validation: validation_ratio,
                                                      SplitNames.test: test_ratio}
        self.seed = seed

    def split(self, dataset_metadata: DatasetMetadata) -> TrainValidationTestSplitIndices:
        return perform_ratio_split(dataset_metadata=dataset_metadata,
                                   split_ratios=self.split_ratios,
                                   seed=self.seed)

    def __str__(self):
        return f"ratio-{self.split_ratios.get(SplitNames.train)}_{self.split_ratios.get(SplitNames.validation)}_" \
               f"{self.split_ratios.get(SplitNames.test)}"


def perform_ratio_split(dataset_metadata: DatasetMetadata,
                        split_ratios: Dict[SplitNames, float],
                        seed: int) -> TrainValidationTestSplitIndices:
    """
    ToDo Document me
    :param split_ratios:
    :param dataset_metadata:
    :param seed
    :return:
    """
    session_index = CsvDatasetIndex(dataset_metadata.session_index_path)

    num_samples = len(session_index)
    sample_indices = list(range(num_samples))

    rng = np.random.default_rng(seed)
    rng.shuffle(sample_indices)

    num_samples = len(sample_indices)
    remainder = sample_indices
    splits = dict()
    for split_name, ratio in split_ratios.items():
        remainder_length = len(remainder)
        num_samples_in_split = int(math.ceil(ratio * num_samples))

        # take only what is left for the last split to avoid errors
        num_samples_in_split = min(num_samples_in_split, remainder_length)

        samples = remainder[:num_samples_in_split]
        remainder = remainder[num_samples_in_split:]

        splits[split_name] = samples

    return TrainValidationTestSplitIndices(train_indices=splits[SplitNames.train],
                                           validation_indices=splits[SplitNames.validation],
                                           test_indices=splits[SplitNames.test])
