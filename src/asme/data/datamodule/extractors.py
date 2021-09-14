from abc import abstractmethod
from typing import Dict, Any, Iterable

from asme.data.datasets import ITEM_SEQ_ENTRY_NAME


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
    This TargetPositionExtractor returns all indices between [window_size - 1; len(sequence) - session_end_offset].
    """
    def __init__(self, window_size: int, session_end_offset: int):
        self.window_size = window_size
        self.session_end_offset = session_end_offset

    def apply(self, session: Dict[str, Any]) -> Iterable[int]:
        sequence = session[ITEM_SEQ_ENTRY_NAME]
        return range(self.window_size - 1, len(sequence)- self.session_end_offset)
