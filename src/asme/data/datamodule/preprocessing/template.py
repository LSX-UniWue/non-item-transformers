from pathlib import Path
from typing import List

from asme.data.datamodule.extractors import RemainingSessionPositionExtractor, SlidingWindowPositionExtractor
from asme.data.datamodule.preprocessing.action import PreprocessingActionGroup
from asme.data.datamodule.preprocessing.csv import GroupAndFilter, GroupedFilter, CopyMainFile
from asme.data.datamodule.preprocessing.indexing import CreateSessionIndex, CreateNextItemIndex, \
    CreateSlidingWindowIndex
from asme.data.datamodule.preprocessing.popularity import CreatePopularity
from asme.data.datamodule.preprocessing.split import CreateRatioSplit, CreateLeaveOneOutSplit, \
    CreateLeavePercentageOutSplit
from asme.data.datamodule.preprocessing.vocabulary import CreateVocabulary
from asme.data.datasets.sequence import MetaInformation


def build_ratio_split(columns: List[MetaInformation], file_prefix: str,
                      min_item_feedback: int, min_item_feedback_column: str,
                      min_sequence_length: int, min_sequence_length_column: str,
                      session_key_columns: List[str], index_columns: List[MetaInformation],
                      train_percentage: float, validation_percentage: float,
                      test_percentage: float, window_markov_length: int,
                      window_target_length: int, window_session_end_offset: int) -> PreprocessingActionGroup:
    actions = [CopyMainFile(),
               GroupAndFilter("items_filtered", min_item_feedback_column,
                              GroupedFilter("count", lambda v: v >= min_item_feedback)),
               GroupAndFilter("sessions_filtered", min_sequence_length_column,
                              GroupedFilter("count", lambda v: v >= min_sequence_length)),
               CreateSessionIndex(session_key_columns),
               CreateRatioSplit(train_percentage, validation_percentage, test_percentage,
                                per_split_actions=
                                [CreateSessionIndex(session_key_columns),
                                 CreateNextItemIndex(
                                     index_columns,
                                     RemainingSessionPositionExtractor(
                                         min_sequence_length)),
                                 CreateSlidingWindowIndex(
                                     index_columns,
                                     SlidingWindowPositionExtractor(window_markov_length,
                                                                    window_target_length,
                                                                    window_session_end_offset))
                                 ],
                                complete_split_actions=
                                [CreateVocabulary(columns, prefixes=[file_prefix]),
                                 CreatePopularity(columns, prefixes=[file_prefix])])]

    return PreprocessingActionGroup(
        actions,
        name=f"Generating ratio split. ({train_percentage}/{validation_percentage}/{test_percentage}).",
        relative_output_dir=Path(f"./ratio_split-{train_percentage}_{validation_percentage}_{test_percentage}/")
    )


def build_leave_one_out_split(columns: List[MetaInformation], file_prefix: str,
                              min_item_feedback: int, min_item_feedback_column: str,
                              min_sequence_length: int, min_sequence_length_column: str,
                              session_key_columns: List[str], item_column: MetaInformation,
                              index_columns: List[MetaInformation]) -> PreprocessingActionGroup:
    actions = [
        CopyMainFile(),
        GroupAndFilter("items_filtered", min_item_feedback_column,
                       GroupedFilter("count", lambda v: v >= min_item_feedback)),
        GroupAndFilter("sessions_filtered", min_sequence_length_column,
                       GroupedFilter("count", lambda v: v >= min_sequence_length)),
        CreateSessionIndex(session_key_columns),
        CreateLeaveOneOutSplit(item_column,
                               inner_actions=
                               [CreateNextItemIndex(
                                   index_columns,
                                   RemainingSessionPositionExtractor(
                                       min_sequence_length)),
                                   CreateVocabulary(columns),
                                   CreatePopularity(columns)])
    ]

    return PreprocessingActionGroup(
        actions,
        name="Generating leave-one-out split.",
        relative_output_dir=Path("./loo/")
    )


def build_leave_percentage_out_split(columns: List[MetaInformation], file_prefix: str,
                                     min_item_feedback: int, min_item_feedback_column: str,
                                     min_sequence_length: int, min_sequence_length_column: str,
                                     session_key_columns: List[str], item_column: MetaInformation,
                                     index_columns: List[MetaInformation], train_percentage: float,
                                     validation_percentage: float, test_percentage: float,
                                     min_train_length: int, min_validation_length: int,
                                     min_test_length: int) -> PreprocessingActionGroup:
    actions = [
        CopyMainFile(),
        GroupAndFilter("items_filtered", min_item_feedback_column,
                       GroupedFilter("count", lambda v: v >= min_item_feedback)),
        GroupAndFilter("sessions_filtered", min_sequence_length_column,
                       GroupedFilter("count", lambda v: v >= min_sequence_length)),
        CreateSessionIndex(session_key_columns),
        CreateLeavePercentageOutSplit(item_column, train_percentage, validation_percentage,
                                      test_percentage, min_train_length, min_validation_length, min_test_length,
                                      inner_actions=
                                      [CreateNextItemIndex(
                                          index_columns,
                                          RemainingSessionPositionExtractor(
                                              min_sequence_length)),
                                          CreateVocabulary(columns),
                                          CreatePopularity(columns)])
    ]

    return PreprocessingActionGroup(
        actions,
        name=f"Generating leave-percentage-out split ({train_percentage}/{validation_percentage}/{test_percentage}).",
        relative_output_dir=Path(f"./lpo_split-{train_percentage}_{validation_percentage}_{test_percentage}/")
    )
