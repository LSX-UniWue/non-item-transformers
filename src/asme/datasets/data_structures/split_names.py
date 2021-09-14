import enum

TRAIN_KEY = "train"
VALIDATION_KEY = "validation"
TESTING_KEY = "test"


class SplitNames(enum.Enum):
    train = TRAIN_KEY
    validation = VALIDATION_KEY
    test = TESTING_KEY

    def __str__(self):
        return self.value
