from typing import List
from asme.datasets.data_structures.split_names import SplitNames


class TrainValidationTestSplitIndices:
    def __init__(self, train_indices: List[int], validation_indices: List[int], test_indices: List[int]):
        self.train_indices = train_indices
        self.validation_indices = validation_indices
        self.test_indices = test_indices

    def items(self):
        return [(SplitNames.train, self.train_indices),
                (SplitNames.validation, self.validation_indices),
                (SplitNames.test, self.test_indices)]

    def get(self, split_name: SplitNames):
        if split_name == SplitNames.train:
            return self.train_indices
        elif split_name == SplitNames.validation:
            return self.validation_indices
        elif split_name == SplitNames.test:
            return self.test_indices
        else:
            raise Exception("Provided wrong key to TrainValidationTestSplitIndices!")
