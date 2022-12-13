from typing import List

from asme.data.datasets.processors.processor import Processor
from torch.utils.data import Dataset


class PlaceholderDataset(Dataset):

    def __init__(self,
                 processors: List[Processor] = None
                 ):
        super().__init__()
        if processors is None:
            processors = []
        self._processors = processors

    def __len__(self):
        return 1
