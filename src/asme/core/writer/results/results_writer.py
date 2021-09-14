import csv
import json
from abc import abstractmethod
from pathlib import Path
from typing import IO, Dict


class ResultWriter:
    """
    Writer interface for all writers that serialize predictions
    """

    def __init__(self,
                 file_handle: IO[str]
                 ):
        """
        constructor for prediction writers
        :param file_handle: the handle for the file
        """
        super().__init__()
        self.file_handle = file_handle

    @abstractmethod
    def write_overall_results(self, recommender_name: str, metrics: Dict[str, float]):
        """
        writes the metrics calculated over all data points to the file
        :param recommender_name:
        :param metrics:
        :return:
        """
        pass


class JSONResultWriter(ResultWriter):

    """
    The JSONResultWriter writes the results (metrics) into a file encoded as JSON.

    """

    def __init__(self,
                 file_handle: IO[str]):
        super().__init__(file_handle)

    def write_overall_results(self, recommender_name: str, metrics: Dict[str, float]):
        json_object = {
            'recommender_id': recommender_name,
            'metrics': metrics
        }

        json.dump(json_object, self.file_handle)


class CSVResultWriter(ResultWriter):

    """
    The CSVPredictionWriter writes results into a file encoded as CSV.

    Each metric is written as "metric_name, metric_value"

    the first line of the csv contains the recommender id
    """

    HEADER = ["metric name", "value"]

    def __init__(self,
                 file_handle: IO[str]
                 ):
        super().__init__(file_handle)

        self.csv_writer = csv.writer(file_handle)
        self.csv_writer.writerow(self.HEADER)

    def write_overall_results(self, recommender_name: str, metrics: Dict[str, float]):
        metrics_to_write = [[metric_name, metric_value] for metric_name, metric_value in metrics.items()]
        rows_to_write = metrics_to_write + [["recommender_id", recommender_name]]
        self.csv_writer.writerows(rows_to_write)


# for lazy loading only point to the class
SUPPORTED_RESULT_WRITERS = {
    '.csv': CSVResultWriter,
    '.json': JSONResultWriter
}


def check_file_format_supported(output_file: Path) -> bool:
    """
    Checks whether the file extension of the provided file corresponds to a supported file format.

    :param output_file: The path to the output file.
    :return: True, if the file format indicated by the output file is supported. False otherwise.
    """
    file_extension = output_file.suffix
    return file_extension in SUPPORTED_RESULT_WRITERS


def build_result_writer(file_handle: IO[str]
                        ) -> ResultWriter:
    """
    Builds the result writer for the specified file handle based on the extension of the underlying file of the
    file_handle

    Currently the following file extensions are supported:

    - json
    - csv

    :param file_handle: the file handle to use
    :return:
    """
    output_file_path = Path(file_handle.name)
    file_extension = output_file_path.suffix

    if not check_file_format_supported(output_file_path):
        raise KeyError(f'{file_extension} is not a supported format to write predictions to file')
    return SUPPORTED_RESULT_WRITERS[file_extension](file_handle)
