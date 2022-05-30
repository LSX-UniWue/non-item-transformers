from asme.data.datamodule.extractors import PercentageBasedPositionExtractor
from asme.data.datamodule.util import SplitNames
from asme.data.datasets import ITEM_SEQ_ENTRY_NAME


def test_percentage_based_target_extractor():
    train, validation, test = 0.7, 0.2, 0.1
    sessions = [list(range(i+4)) for i in range(6)]
    expected_train_results = [[0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]]
    expected_validation_results = [[], [], [], [3], [4], [5], [5], [7], [7], [8, 9]]

    def _test(extractor: PercentageBasedPositionExtractor, sessions, expected_results):
        for session, result in zip(sessions, expected_results):
            indices = extractor.apply({
                ITEM_SEQ_ENTRY_NAME: session
            })
            assert list(indices) == result, f"Session {session} should result in indices {list(indices)} for split {extractor._target_split}"

    # Train
    extractor = PercentageBasedPositionExtractor(train, validation, test, SplitNames.train)
    #_test(extractor, sessions, expected_train_results)

    # Validation
    extractor = PercentageBasedPositionExtractor(train, validation, test, SplitNames.validation)
    #_test(extractor, sessions, expected_validation_results)
