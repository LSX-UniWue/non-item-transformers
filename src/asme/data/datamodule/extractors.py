import math
from abc import abstractmethod
from typing import Dict, Any, Iterable, TypeVar, Callable, List, Tuple

from asme.data.datamodule.util import SplitNames, approx_equal
from asme.data.datasets import ITEM_SEQ_ENTRY_NAME

T = TypeVar("T")


class TargetPositionExtractor:
    """
    Base class for different strategies regarding the extraction of target indices from a session.
    """

    @abstractmethod
    def apply(self, session: Dict[str, Any]) -> Iterable[int]:
        """
        Extracts a set of target indices from the session.

        :param session: A dictionary containing the information of a single session.
        """
        pass

    def __call__(self, session: Dict[str, Any]) -> Iterable[int]:
        return self.apply(session)


class FixedOffsetPositionExtractor(TargetPositionExtractor):
    """
    This TargetPositionExtractor always returns the index exactly offset positions away from the end of the session.
    """

    def __init__(self, offset: int):
        """
        :param offset: The offset of the target position from the end of the sequence. This offset ranges from 1 for the
        last item in the sequence to sequence_length for the second item in the sequence.
        """
        self.offset = offset

    def apply(self, session: Dict[str, Any]) -> Iterable[int]:
        return [len(session[ITEM_SEQ_ENTRY_NAME]) - self.offset - 1]


class RemainingSessionPositionExtractor(TargetPositionExtractor):
    """
    This TargetPositionExtractor returns all indices between min_session_length and sequence_length
    """

    def __init__(self, min_session_length: int = 1):
        """
        :param min_session_length: The minimum length of a subsequence of the session that has to be reached before
        extracting targets begins. For instance, min_session_length = 5 would yield 5 (i.e the sixth entry) as the first
        target index.
        """
        self.min_session_length = min_session_length

    def apply(self, session: Dict[str, Any]) -> Iterable[int]:
        return range(self.min_session_length, len(session[ITEM_SEQ_ENTRY_NAME]))


class SlidingWindowPositionExtractor(TargetPositionExtractor):
    """
    The SlidingWindowPositionExtractor returns all indices between [window_size - 1; len(sequence) - session_end_offset]

    :param window_markov_length: The length of the markov order of each window.
                                (Each window consists of the markov order and a number of targets)
    :param window_target_length: Indicates the target size for a window
    :param session_end_offset: Indicates how many items are cut off from the right



    """

    def __init__(self, window_markov_length: int, window_target_length: int, session_end_offset: int):
        self.window_size = window_markov_length + window_target_length
        self.window_target_length = window_target_length
        self.session_end_offset = session_end_offset

    def apply(self, session: Dict[str, Any]) -> Iterable[int]:
        sequence = session[ITEM_SEQ_ENTRY_NAME]
        # the sequence will later be left_padded if its length is shorter than the window_size
        start = self.window_size - 1 if self.window_size <= len(sequence) - self.session_end_offset \
            else len(sequence) - self.session_end_offset - self.window_target_length
        return range(start, len(sequence) - self.session_end_offset)


class PercentageBasedPositionExtractor(TargetPositionExtractor):
    """
    Returns the indices of the items of a session that should belong to the given target split using the provided
    percentages.

    :param train_percentage: Fraction of items of the session to use as training data.
    :param validation_percentage: Fraction of items of the session to use as validation data.
    :param test_percentage: Fraction of items of the session to use as test data.
    :param target_split: The split for which the indices should be extracted.
    """

    def __init__(self, train_percentage: float, validation_percentage: float, test_percentage: float, target_split: SplitNames,
                 min_train_length: int = 2, min_validation_length: int = 1, min_test_length: int = 1):
        self._train_percentage = train_percentage
        self._validation_percentage = validation_percentage
        self._test_percentage = test_percentage
        self._min_test_length = min_test_length
        self._min_validation_length = min_validation_length
        self._min_train_length = min_train_length
        self._target_split = target_split

    def apply(self, session: Dict[str, Any]) -> Iterable[int]:
        sequence = session[ITEM_SEQ_ENTRY_NAME]
        num_train_items, num_validation_items, num_test_items = self._compute_items_per_split(sequence)
        if self._target_split is SplitNames.train:
            return range(num_train_items)
        elif self._target_split is SplitNames.validation:
            return range(num_train_items, num_train_items + num_validation_items)
        elif self._target_split is SplitNames.test:
            return range(num_train_items + num_validation_items, num_train_items + num_validation_items + num_test_items)

    def _compute_items_per_split(self, sequence) -> Tuple[int, int, int]:
        """
        This method computes the number of items of the given sequence that should belong to training, validation and
        test split.
        """
        EPS = 1e-10

        def _fractional_part(num: float) -> float:
            return num - int(num)

        def _first_index(seq: Iterable[T], condition: Callable[[T], bool]) -> int:
            try:
                return next(i for i, x in enumerate(seq) if condition(x))
            except StopIteration:
                return -1

        def _satisfies_hard_constraints(num_items: List[float]) -> bool:
            return num_items[0] >= 2 and num_items[1] >= 1 and num_items[2] >= 1

        def _hard_constraint_differences(num_items: List[float]) -> List[float]:
            return [x - c for x, c in zip(num_items,
                                          [self._min_train_length, self._min_validation_length, self._min_test_length])]

        def _correct_for_floating_point_precision(num_items: List[float]) -> List[float]:
            corrected = []
            for i in range(len(num_items)):
                x = num_items[i]
                if _fractional_part(x) < EPS:
                    corrected.append(math.floor(x))
                elif _fractional_part(x) > (1 - EPS):
                    corrected.append(math.ceil(x))
                else:
                    corrected.append(x)

            return corrected

        def _integerize(num_items: List[float]) -> List[int]:
            """
            This method "rounds" items counts to ints by moving the fractional parts towards the training set.
            """
            train_fractional_part = _fractional_part(num_items[0])
            validation_fractional_part = _fractional_part(num_items[1])
            test_fractional_part = _fractional_part(num_items[2])

            cumulative_fractional_part = _fractional_part(train_fractional_part + validation_fractional_part + test_fractional_part)
            assert approx_equal(cumulative_fractional_part, 0.0, EPS) or approx_equal(cumulative_fractional_part, 1.0, EPS)
            precision_corrected_counts = _correct_for_floating_point_precision(
                [num_items[0] + validation_fractional_part + test_fractional_part,
                 num_items[1] - validation_fractional_part,
                 num_items[2] - test_fractional_part])

            return [int(x) for x in precision_corrected_counts]

        length = len(sequence)

        # Ensure all sessions have enough items to satisfy the constraints on train, validation and test set size.
        required_min_session_length = self._min_train_length + self._min_validation_length + self._min_test_length
        if length < required_min_session_length:
            raise ValueError(f"Encountered a session of length {length} < {required_min_session_length}. "
                             f"These sessions can not be split correctly and should be removed during preprocessing.")

        optimal_item_counts = [length * self._train_percentage,
                               length * self._validation_percentage,
                               length * self._test_percentage]

        # Check if all constraints are fulfilled
        if _satisfies_hard_constraints(optimal_item_counts):
            return tuple(_integerize(optimal_item_counts))

        # At this point, at least one constraint is "over-fulfilled" since at least one is not fulfilled but
        # length >= 4
        adjusted_item_counts = optimal_item_counts
        # As long as the constraints are not fulfilled, we transfer as much items from a fulfilled constraint to an
        # unfulfilled one as possible without violating the first one again
        while not _satisfies_hard_constraints(adjusted_item_counts):
            hard_constraint_differences = _hard_constraint_differences(adjusted_item_counts)
            fulfilled_index = _first_index(hard_constraint_differences, lambda x: x > 0)
            unfulilled_index = _first_index(hard_constraint_differences, lambda x: x < 0)
            # We need to make sure not to violate the currently over-fulfilled constraint again!
            delta = min([hard_constraint_differences[fulfilled_index],
                         abs(hard_constraint_differences[unfulilled_index])])
            adjusted_item_counts[fulfilled_index] -= delta
            adjusted_item_counts[unfulilled_index] += delta
            adjusted_item_counts = _correct_for_floating_point_precision(adjusted_item_counts)

        return tuple(_integerize(adjusted_item_counts))
