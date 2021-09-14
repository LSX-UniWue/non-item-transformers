import torch


def get_true_positives(prediction: torch.Tensor,
                       positive_item_mask: torch.Tensor,
                       k: int,
                       metric_mask: torch.Tensor
                       ) -> torch.Tensor:
    """
    returns the mask of which items are relevant
    :param prediction:
    :param positive_item_mask:
    :param k:
    :param metric_mask:
    :return:
    """
    # here we mask the padding tokens with the lowest possible value
    prediction = prediction.masked_fill(metric_mask == 0, torch.finfo(torch.float).min)

    # get the indices of the top k predictions
    predicted_id = torch.argsort(prediction, descending=True)

    # limit the prediction to the top k
    predicted_id = predicted_id[:, :k]

    # select the mask for each of the indices, this is than the tp
    return positive_item_mask.gather(1, predicted_id)


def get_ranks(prediction: torch.Tensor,
              positive_item_mask: torch.Tensor,
              metrics_mask: torch.Tensor
              ) -> torch.Tensor:
    """
    returns the rank of the first relevant item
    :param prediction: the prediction logtis :math`(N, I)`
    :param positive_item_mask: a mask where 1 at position i indicates that the item at index i is relevant :math`(N, I)`
    :param metrics_mask: the metrics mask
    :return:
    """
    device = prediction.device
    num_positions = prediction.size()[1] + 1
    tp = get_true_positives(prediction, positive_item_mask, num_positions, metrics_mask)
    ranks = torch.arange(1, num_positions).unsqueeze(0).repeat(prediction.size()[0], 1).to(device=device)
    rank = (ranks * tp).max(dim=-1).values
    return rank


def get_true_positive_count(prediction: torch.Tensor,
                            positive_item_mask: torch.Tensor,
                            k: int,
                            metrics_mask: torch.Tensor
                            ) -> torch.Tensor:
    """
    returns the number of true positives for
    :param k: k for the @k metric
    :param prediction: the prediction logits :math`(N, I)`
    :param positive_item_mask: a mask where 1 at position i indicates that the item at index i is relevant :math`(N, I)`
    :param metrics_mask: the metrics mask
    :return: the number of true positives (N)
    """
    tp = get_true_positives(prediction, positive_item_mask, k, metrics_mask)
    return tp.sum(1)  # get the total number of positive items


def calc_recall(prediction: torch.Tensor,
                positive_item_mask: torch.Tensor,
                k: int,
                metric_mask: torch.Tensor
                ) -> torch.Tensor:
    """
    calcs the recall given the predictions and postive item mask
    :param positive_item_mask: a mask, where 1 at index i indicates that item i in predictions is relevant
    :param prediction: the prediction logits :math`(N, I)
    :param k: the k
    :param metric_mask: the metrics mask
    :return: the recall :math`(N)`

    where N is the batch size and I the number of items to consider
    """
    tp = get_true_positive_count(prediction, positive_item_mask, k, metric_mask)
    all_relevant_items = positive_item_mask.sum(dim=1)
    recall = tp / all_relevant_items
    # maybe there are no relevant items
    recall[torch.isnan(recall)] = 0
    return recall


def calc_precision(prediction: torch.Tensor,
                   positive_item_mask: torch.Tensor,
                   k: int,
                   metric_mask: torch.Tensor
                   ) -> torch.Tensor:
    """
    calcs the precision given the predictions and positive item mask
    :param positive_item_mask: a mask, where 1 at index i indicates that item i in predictions is relevant
    :param prediction: the prediction logits :math`(N, I)
    :param k: the k
    :param metric_mask: the metrics mask
    :return: the precision :math`(N)`

    where N is the batch size and I the number of items to consider
    """

    tp = get_true_positive_count(prediction, positive_item_mask, k, metric_mask)
    return tp / k


def _build_dcg_values(end: int,
                      batch_size: int
                      ) -> torch.Tensor:
    steps = torch.arange(2, end + 2).to(dtype=torch.float)

    dcg_values = 1 / torch.log2(steps)
    return dcg_values.unsqueeze(0).repeat(batch_size, 1)


def calc_ndcg(prediction: torch.Tensor,
              positive_item_mask: torch.Tensor,
              k: int,
              metric_mask: torch.Tensor
              ) -> torch.Tensor:
    """
    calculates the NDCG given the predictions and positive item mask
    :param metric_mask: the metrics mask
    :param positive_item_mask: a mask, where 1 at index i indicates that item i in predictions is relevant
    :param prediction: the prediction logits :math`(N, I)
    :param k: the k
    :return: the ndcg :math`(N)`

    where N is the batch size and I the number of items to consider
    """

    dcg = calc_dcg(prediction, positive_item_mask, k, metric_mask)

    idcg_all = _build_dcg_values(k, positive_item_mask.size()[0]).to(prediction.device)

    # sum all relevant items per batch
    number_relevant_items = positive_item_mask.sum(dim=1).to(dtype=torch.int64)
    # restrict them to k, there could be more
    number_relevant_items = torch.where(number_relevant_items <= k, number_relevant_items, k)
    number_relevant_items = number_relevant_items.unsqueeze(1).repeat(1, k)
    # only use calculated values of the dcg that are lt than the max relevant items
    relevant_mask = torch.arange(0, k).type_as(prediction).lt(number_relevant_items)
    idcg = idcg_all * relevant_mask

    ndcg = dcg / idcg.sum(dim=1)
    ndcg[torch.isnan(ndcg)] = 0
    return ndcg


def calc_dcg(prediction: torch.Tensor,
             positive_item_mask: torch.Tensor,
             k: int,
             metric_mask: torch.Tensor
             ) -> torch.Tensor:
    """
    calculates the DCG of the prediction given the positive item mask and k
    :param metric_mask: the metric mask to
    :param prediction: the prediction (N, I)
    :param positive_item_mask: the positive item mask (N, I)
    :param k: the k
    :return: the dcg (N)

    where N is the batch size and I the number of items
    """
    device = prediction.device

    tp = get_true_positives(prediction, positive_item_mask, k, metric_mask)

    num_items = min(k, prediction.size()[1])

    dcg_values = _build_dcg_values(num_items, positive_item_mask.size()[0]).to(device=device)
    dcg = dcg_values * tp
    return dcg.sum(dim=1)
