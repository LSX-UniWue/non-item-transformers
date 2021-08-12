import torch

import torch.nn.functional as F

from asme.losses.losses import SequenceRecommenderContrastiveItemLoss
from asme.tokenization.tokenizer import Tokenizer


class DreamContrastiveLoss(SequenceRecommenderContrastiveItemLoss):

    """
    a loss for basket recommendation using a bpr like loss function
    """

    def __init__(self,
                 item_tokenizer: Tokenizer):
        super().__init__()
        self.item_tokenizer = item_tokenizer

    def forward(self,
                logit: torch.Tensor,
                positive_items: torch.Tensor,
                negative_items: torch.Tensor,
                ) -> torch.Tensor:
        # bpr FIXME: check
        # we only use the last position as target, because the rnn only encodes the complete sequence
        padding_mask = (~ positive_items.eq(self.item_tokenizer.pad_token_id)).max(-1).values.sum(-1) - 1
        target_mask = F.one_hot(padding_mask, num_classes=positive_items.size()[1]).to(torch.bool)

        pos_items = positive_items[target_mask]
        neg_items = negative_items[target_mask]
        pos_logits = logit.gather(1, pos_items)
        neg_logits = logit.gather(1, neg_items)

        mask = ~ pos_items.eq(self.item_tokenizer.pad_token_id)
        num_items = mask.sum()

        score = F.logsigmoid(pos_logits - neg_logits)
        score = score * mask

        return - score.sum() / num_items
