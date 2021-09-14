import torch
from torch.nn.parameter import Parameter

from asme.data.datasets import POSITIVE_SAMPLES_ENTRY_NAME, NEGATIVE_SAMPLES_ENTRY_NAME
from asme.core.metrics.container.metrics_container import MetricsContainer
from asme.core.modules.metrics_trait import MetricsTrait
from pytorch_lightning import core as pl
from torch.nn import functional as F

from asme.core.tokenization.tokenizer import Tokenizer


class BprModule(MetricsTrait, pl.LightningModule):

    def __init__(self,
                 item_tokenizer: Tokenizer,
                 user_tokenizer: Tokenizer,
                 embedding_size: int,
                 regularization_factor: float,
                 metrics: MetricsContainer):
        super().__init__()
        self.item_tokenizer = item_tokenizer
        self.user_tokenizer = user_tokenizer
        self.embedding_size = embedding_size
        self.num_users = len(user_tokenizer)
        self.num_items = len(item_tokenizer)
        self.regularization_factor = regularization_factor
        self.metrics = metrics

        self.W = Parameter(torch.zeros([self.num_users, self.embedding_size]))
        self.H = Parameter(torch.zeros([self.num_items, self.embedding_size]))

    def get_metrics(self) -> MetricsContainer:
        return self.metrics
    
    def forward(self,
                user: torch.Tensor,
                item: torch.Tensor
                ):
        """
        Computes the score of a user-item pair by calculating the dot product of the corresponding rows in the
        user matrix W and item matrix H
        :param user: A tensor of shape (N,I) containing user ids.
        :param item: A tensor of shape (N,I) containing item ids.
        :returns: A tensor of shape (N,1) containing the scores of the user-item pairs provided.
        """
        u_emb = self.W.index_select(dim=0, index=user)
        i_emb = self.H.index_select(dim=0, index=item)

        x_ui = torch.sum(u_emb * i_emb, dim=-1)

        return x_ui

    def calculate_regularization_penalty(self, x_ui: torch.Tensor, x_uj: torch.Tensor) -> torch.Tensor:
        return self.regularization_factor * (x_ui.abs().mean() + x_uj.abs().mean())

    def training_step(self, batch, batch_idx):
        user = batch["user_id"]
        item_i = batch[POSITIVE_SAMPLES_ENTRY_NAME]
        item_j = batch[NEGATIVE_SAMPLES_ENTRY_NAME]

        x_ui = self.forward(user, item_i)
        x_uj = self.forward(user, item_j)

        x_uij = x_ui - x_uj

        loss = -F.logsigmoid(x_uij).mean() + self.calculate_regularization_penalty(x_ui, x_uj)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def shared_step(self, batch, batch_idx) -> torch.Tensor:
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


