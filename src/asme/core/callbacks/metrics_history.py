from pytorch_lightning import Callback


class MetricsHistoryCallback(Callback):
    """
    Captures the reported metrics after every validation epoch.
    """

    def __init__(self):
        super().__init__()

        self.metric_history = []

    def on_validation_end(self, pl_trainer, pl_module):
        self.metric_history.append(pl_trainer.callback_metrics)
