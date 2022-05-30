import csv
from abc import abstractmethod
from typing import List, Optional, IO, Tuple
import itertools

from asme.core.evaluation.evaluation import BatchEvaluator


def all_equal_in_list(lst):
    return not lst or lst.count(lst[0]) == len(lst)

#TODO: Maybe add JSONWriters

class BatchEvaluationWriter:

    def __init__(self, evaluators: List[BatchEvaluator]):
        self.evaluators = evaluators

    @abstractmethod
    def write_evaluation(self,
                         batch_index,
                         batch,
                         logits):
        pass

    @abstractmethod
    def init_file(self, file_handle: IO[str]):
        pass


class CSVMultiLineWriter(BatchEvaluationWriter):
    """
    The EvaluationWriter writes predictions into a file encoded as CSV, taking one batch at a time.

    Each sample is written to the file in the
    following way:

    For all samples each recommendation will be written in a separate line. The positions of the recommendation is added
    as a field in the CSV line to keep the information where in the recommendation list the recommendation item(s)
    was/were.
    """

    def __init__(self, evaluators: List[BatchEvaluator]):
        super().__init__(evaluators)
        self.headers = [["order"]] + [evaluator.get_header() for evaluator in evaluators]

    def init_file(self, file_handle: IO[str]):
        self.file_handle = file_handle
        self.csv_writer = csv.writer(file_handle)
        self.csv_writer.writerow(itertools.chain.from_iterable(self.headers))

    def write_evaluation(self,
                         batch_index,
                         batch,
                         logits):

        results = [(evaluator.evaluate(batch_index, batch, logits), evaluator.eval_samplewise(), evaluator.get_header())
                   for evaluator in self.evaluators]

        # Per Sample in batch
        rows_to_write = []
        for sample in range(logits.shape[0]):
            num_rows = [len(eval[1]) for eval, samplewise, header in results if not samplewise]
            # if not all_equal_in_list(num_rows):
            #    raise ValueError(f"Evaluators output size must be of same length, but is {num_rows}.")
            for i in range(num_rows[0]):
                row_to_write = [str(i + 1)]
                for eval, samplewise, header in results:
                    res = eval[sample] if samplewise else eval[sample][i]
                    if len(header) > 1:
                        row_to_write.extend(res)
                    else:
                        row_to_write.append(res)

                rows_to_write.append(row_to_write)
        self.csv_writer.writerows(rows_to_write)

class CSVSingleLineWriter(BatchEvaluationWriter):
    """
    The EvaluationWriter writes predictions into a file encoded as CSV, taking one batch at a time.

    Each sample is written to the file in the
    following way:

    For each sample all recommendations will be written in one line in ordered lists.
    """

    def __init__(self, evaluators: List[BatchEvaluator]):
        super().__init__(evaluators)
        self.headers = [evaluator.get_header() for evaluator in evaluators]

    def init_file(self, file_handle: IO[str]):
        self.file_handle = file_handle
        self.csv_writer = csv.writer(file_handle)
        self.csv_writer.writerow(itertools.chain.from_iterable(self.headers))

    def write_evaluation(self,
                         batch_index,
                         batch,
                         logits):

        results = [(evaluator.evaluate(batch_index, batch, logits), evaluator.eval_samplewise(), evaluator.get_header())
                   for evaluator in self.evaluators]

        # Per Sample in batch
        rows_to_write = []
        for sample in range(logits.shape[0]):
            row_to_write = []
            for eval, samplewise, header in results:
                res = eval[sample]
                if len(header) > 1:
                    row_to_write.extend(res)
                else:
                    row_to_write.append(res)

            rows_to_write.append(row_to_write)
        self.csv_writer.writerows(rows_to_write)
