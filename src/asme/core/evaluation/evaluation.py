from abc import abstractmethod
from typing import Dict, Any, List

import numpy as np
import torch
from asme.core.evaluation.pred_utils import _extract_sample_metrics, get_positive_item_mask
from asme.core.metrics.metric import MetricStorageMode
from asme.data.datasets import ITEM_SEQ_ENTRY_NAME, SAMPLE_IDS, TARGET_ENTRY_NAME, SESSION_IDENTIFIER
from torch import Tensor


class BatchEvaluator:
    """
    Evaluator for batch level evaluation
    """

    @abstractmethod
    def evaluate(self,
                 batch_index: int,
                 batch: Dict[str, Any],
                 logits: Tensor,
                 ) -> List[Any]:
        """
        Execute evaluation on batch and processed logits.

        :param batch_index:
        :param batch:
        :param logits:

        :return:
        """
        pass

    @abstractmethod
    def get_header(self) -> List[str]:
        pass

    @abstractmethod
    def eval_samplewise(self) -> bool:
        pass


class LogInputEvaluator(BatchEvaluator):
    """
    Extract the original input sequence
    """

    def __init__(self, item_tokenizer):
        self.item_tokenizer = item_tokenizer
        self.header = ["input"]
        tokens = np.asarray(self.item_tokenizer.vocabulary.tokens())
        ids = np.asarray(self.item_tokenizer.vocabulary.ids())
        self.vocab_lookup = np.empty(max(ids) + 1, dtype=tokens.dtype)
        self.vocab_lookup[ids] = tokens
        self.special_tokens = np.asarray(self.item_tokenizer.get_special_token_ids())

    def get_header(self) -> List[str]:
        return self.header

    def eval_samplewise(self) -> bool:
        return True

    def evaluate(self,
                 batch_index: int,
                 batch: Dict[str, Any],
                 logits: Tensor,
                 ) -> List[Any]:
        input_ids = np.asarray(batch[ITEM_SEQ_ENTRY_NAME])
        input_tokens = self.vocab_lookup[input_ids]

        filter_special_tokens_boolean = np.isin(input_ids, self.special_tokens, invert=True)
        filtered_result = [input_tokens[sample][filter_special_tokens_boolean[sample]].tolist() for sample in
                           range(input_tokens.shape[0])]

        return filtered_result


class ExtractSampleIdEvaluator(BatchEvaluator):
    """
    Create Sample ID based on SESSION_IDENTIFIER or sample ID
    """

    def __init__(self, use_session_id: bool = False):
        self.header = ["SID"]
        self.use_session_id = use_session_id

    def get_header(self) -> List[str]:
        return self.header

    def eval_samplewise(self) -> bool:
        return True

    def evaluate(self,
                 batch_index: int,
                 batch: Dict[str, Any],
                 logits: Tensor
                 ) -> List[Any]:

        if self.use_session_id:
            sample_ids = batch[SESSION_IDENTIFIER]
        else:
            sample_ids = batch[SAMPLE_IDS].tolist()
        sequence_position_ids = None
        if 'pos' in batch:
            sequence_position_ids = batch['pos'].tolist()
        id_lambda = lambda x: self._generate_sample_id(sample_ids, sequence_position_ids, x)
        sample_ids = [id_lambda(batch_sample) for batch_sample in range(logits.shape[0])]

        return sample_ids

    @staticmethod
    def _generate_sample_id(sample_ids, sequence_position_ids, sample_index) -> str:
        sample_id = sample_ids[sample_index]
        if sequence_position_ids is None:
            return sample_id
        return f'{sample_id}_{sequence_position_ids[sample_index]}'


class TrueTargetEvaluator(BatchEvaluator):
    """
    Extract the true target
    """

    def __init__(self, item_tokenizer):
        self.header = ["target"]
        self.item_tokenizer = item_tokenizer
        tokens = np.asarray(self.item_tokenizer.vocabulary.tokens())
        ids = np.asarray(self.item_tokenizer.vocabulary.ids())
        self.vocab_lookup = np.empty(max(ids) + 1, dtype=tokens.dtype)
        self.vocab_lookup[ids] = tokens

    def get_header(self) -> List[str]:
        return self.header

    def eval_samplewise(self) -> bool:
        return True

    def evaluate(self,
                 batch_index: int,
                 batch: Dict[str, Any],
                 logits: Tensor,
                 ) -> List[Any]:
        targets = batch[TARGET_ENTRY_NAME].numpy()
        targets = self.vocab_lookup[targets].tolist()
        targets = [[target] for target in targets]
        return targets


class ExtractScoresEvaluator(BatchEvaluator):
    """
    Extract the scores for each recommended item.
    Output: List of lists with scores for each recommended item for each sample in batch
    """

    def __init__(self, item_tokenizer, num_predictions: int, selected_items=None):
        self.header = ["score"]
        self.item_tokenizer = item_tokenizer
        self.num_predictions = num_predictions
        self.selected_items = selected_items
        self.filter_items = np.asarray(self.selected_items)

    def get_header(self) -> List[str]:
        return self.header

    def eval_samplewise(self) -> bool:
        return False

    def evaluate(self,
                 batch_index: int,
                 batch: Dict[str, Any],
                 logits: Tensor,
                 ) -> List[Any]:
        softmax = torch.softmax(logits, dim=-1)
        scores, indices = torch.sort(softmax, dim=-1, descending=True)
        scores = scores[:, :self.num_predictions]
        scores = scores.cpu().numpy()

        n_indices = indices[:, :self.num_predictions].cpu().numpy()

        if self.selected_items is not None:
            filter_boolean = np.isin(n_indices, self.filter_items)
            scores = [scores[sample][filter_boolean[sample]].tolist() for sample in range(scores.shape[0])]
        else:
            scores = scores.tolist()

        return scores


class ExtractRecommendationEvaluator(BatchEvaluator):
    """
    Extract the recommendation, always needed.
    Output: List of lists with recommended items for each sample in batch
    """

    def __init__(self, item_tokenizer, num_predictions: int, selected_items=None):
        self.header = ["recommendation"]
        self.item_tokenizer = item_tokenizer
        self.num_predictions = num_predictions
        self.selected_items = selected_items

        tokens = np.asarray(self.item_tokenizer.vocabulary.tokens())
        ids = np.asarray(self.item_tokenizer.vocabulary.ids())
        self.vocab_lookup = np.empty(max(ids) + 1, dtype=tokens.dtype)
        self.vocab_lookup[ids] = tokens

        self.filter_items = np.asarray(self.selected_items)

    def get_header(self) -> List[str]:
        return self.header

    def eval_samplewise(self) -> bool:
        return False

    def evaluate(self,
                 batch_index: int,
                 batch: Dict[str, Any],
                 logits: Tensor,
                 ) -> List[Any]:

        softmax = torch.softmax(logits, dim=-1)
        scores, indices = torch.sort(softmax, dim=-1, descending=True)
        indices = indices[:, :self.num_predictions]

        n_indices = indices.cpu().numpy()
        result = self.vocab_lookup[n_indices]

        if self.selected_items is not None:
            filter_boolean = np.isin(n_indices, self.filter_items)
            result = [result[sample][filter_boolean[sample]].tolist() for sample in range(result.shape[0])]
        else:
            result = result.tolist()

        return result


class PerSampleMetricsEvaluator(BatchEvaluator):
    """
    Calculates the metrics per sample
    """

    def __init__(self, item_tokenizer, selected_items, module):
        self.item_tokenizer = item_tokenizer
        self.selected_items = selected_items
        self.module = module

        metrics_container = module.metrics
        self.header = metrics_container.get_metric_names()
        self.samplewise_metrics_set = False

    def get_header(self) -> List[str]:
        return self.header

    def eval_samplewise(self) -> bool:
        return True

    def evaluate(self,
                 batch_index: int,
                 batch: Dict[str, Any],
                 logits: Tensor,
                 ) -> List[Any]:

        # Can only set it here, to not interfere with training
        if not self.samplewise_metrics_set:
            metrics_container = self.module.metrics
            for metric in metrics_container.get_metrics():
                metric.set_metrics_storage_mode(MetricStorageMode.PER_SAMPLE)
            self.samplewise_metrics_set = True

        targets = batch[TARGET_ENTRY_NAME]
        metrics = _extract_sample_metrics(self.module)

        num_classes = logits.size()[1]

        item_mask = get_positive_item_mask(targets, num_classes).to(logits.device)

        if self.selected_items:
            selected_items_tensor = torch.tensor(self.selected_items, dtype=torch.int32, device=logits.device)
            logits = torch.index_select(logits, 1, selected_items_tensor)
            item_mask = torch.index_select(item_mask, 1, selected_items_tensor)

        for name, metric in metrics:
            metric.update(logits, item_mask)

        metric_values = [value.raw_metric_values()[batch_index].cpu().numpy() for name, value in metrics]
        metric_results = np.asarray(metric_values).T.tolist()

        return metric_results
