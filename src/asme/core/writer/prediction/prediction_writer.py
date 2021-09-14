import csv
import json
from abc import abstractmethod
from pathlib import Path
from typing import List, Union, Optional, IO, Tuple


class PredictionWriter:
    """
    Writer interface for all writers that serialize predictions
    """

    def __init__(self,
                 file_handle: IO[str],
                 log_input: bool
                 ):
        """
        constructor for prediction writers
        :param file_handle: the handle for the file
        :param log_input: true iff the input sequence to the network should be logged
        """
        super().__init__()
        self.file_handle = file_handle
        self.log_input = log_input

    @abstractmethod
    def write_values(self,
                     sample_id: str,
                     recommendations: List[str],
                     scores: List[float],
                     targets: Union[str, List[str]],
                     metrics: List[Tuple[str, float]],
                     input_sequence: Optional[Union[List[str], List[List[str]]]]):
        """
        writes a sample with all recommendations and corresponding scores to the file handle in the format defined by
        the writer
        :param sample_id: the id of the sample
        :param recommendations: the recommended items (list of size N)
        :param scores: the scores (list of size N) where score at index i is the score of the i-th item
        in recommendations
        :param targets: the actual item(s) that should be recommended
        :param metrics: A list of tuples containing metric name and value computed for this sample
        :param input_sequence:
        :return:
        """
        pass


class JSONPredictionWriter(PredictionWriter):

    """
    The JSONPredictionWriter writes predictions into a file encoded as JSON. Each sample is written to the file as a
    JSON object per line.

    A JSON object looks like this:

    >>> {"sample_id": "0_1",
    >>>  "target": "Item 14", "metrics:" {"MMR@1": 0.23, "MMR@3": 0.33}, "recommendations": [{"item": "<MASK>", "score": 0.09834106266498566}, {"item": "Item 1", "score": 0.09308887273073196}, {"item": "<UNK>", "score": 0.08346869796514511}, {"item": "Item 14", "score": 0.08295264840126038}, {"item": "Item 5", "score": 0.08237089216709137}], "input": ["Item 1"]}

    """

    def __init__(self,
                 file_handle: IO[str],
                 log_input: bool):
        super().__init__(file_handle, log_input)

    def write_values(self, sample_id: str,
                     recommendations: List[str],
                     scores: List[float],
                     targets: Union[str, List[str]],
                     metrics: List[Tuple[str, float]],
                     input_sequence: Optional[Union[List[str], List[List[str]]]]
                     ):
        json_object = {
            'sample_id': sample_id,
            'target': targets
        }

        json_recommendations = []
        for recommendation, score in zip(recommendations, scores):
            json_recommendations.append({
                'item': recommendation,
                'score': score
            })

        json_object['recommendations'] = json_recommendations

        json_metrics = {name: value for (name, value) in metrics}
        json_object['metrics'] = json_metrics

        if input_sequence is not None:
            json_object['input'] = input_sequence

        json.dump(json_object, self.file_handle)
        self.file_handle.write('\n')


class CSVPredictionWriter(PredictionWriter):

    """
    The CSVPredictionWriter writes predictions into a file encoded as CSV. Each sample is written to the file in the
    following way:

    For each sample the recommendations will be written in a separate line. The positions of the recommendation is added
    as a field in the CSV line to keep the information where in the recommendation list the recommendation item(s)
    was/were.
    """

    HEADER = ["SID", "RECOMMENDATION_POSITION", "RECOMMENDATION", "PROBABILITY", "TARGET"]
    INPUT_HEADER_NAME = "INPUT"

    def __init__(self,
                 file_handle: IO[str],
                 log_input: bool
                 ):
        super().__init__(file_handle, log_input)

        self.csv_writer = csv.writer(file_handle)
        self._header_written = False

    def write_values(self, sample_id: str,
                     recommendations: List[str],
                     scores: List[float],
                     targets: Union[str, List[str]],
                     metrics: List[Tuple[str, float]],
                     input_sequence: Optional[Union[List[str], List[List[str]]]]):

        def _extract_metric_names(metrics: List[Tuple[str,float]]) -> List[str]:
            return list(map(lambda metric: metric[0], metrics))

        def _extract_metric_values(metrics: List[Tuple[str,float]]) -> List[float]:
            return list(map(lambda metric: metric[1], metrics))

        if not self._header_written:
            header_to_write = self.HEADER
            header_to_write += _extract_metric_names(metrics)
            if self.log_input:
                header_to_write += [self.INPUT_HEADER_NAME]
            self.csv_writer.writerow(header_to_write)
            self._header_written = True

        rows_to_write = []
        metric_row = _extract_metric_values(metrics)
        # loop through the recommendations and scores and build the rows to write
        for recommendation_position, (recommendation, score) in enumerate(zip(recommendations, scores)):
            row_to_write = [sample_id, recommendation_position + 1, recommendation, score, *metric_row, targets]
            if input_sequence is not None:
                row_to_write.append(input_sequence)
            rows_to_write.append(row_to_write)

        self.csv_writer.writerows(rows_to_write)


def build_prediction_writer(file_handle: IO[str],
                            log_input: bool
                            ) -> PredictionWriter:
    """
    Builds the prediction writer for the specified file handle based on the extension of the underlying file of the
    file_handle

    Currently the following file extensions are supported:

    - json
    - csv

    :param file_handle: the file handle to use
    :param log_input:
    :return:
    """
    file_extension = Path(file_handle.name).suffix

    # for lazy loading only point to the class
    supported_prediction_writers = {
        '.csv': CSVPredictionWriter,
        '.json': JSONPredictionWriter
    }

    if file_extension not in supported_prediction_writers:
        raise KeyError(f'{file_extension} is not a supported format to write predictions to file')
    return supported_prediction_writers[file_extension](file_handle, log_input)
