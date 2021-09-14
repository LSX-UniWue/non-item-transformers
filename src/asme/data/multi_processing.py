import inspect
from abc import abstractmethod

import torch


class MultiProcessSupport:
    """
        Trait that is used to enable support for object recursive multi processing worker initialization in Dataset
        loading related classes.
    """
    def init_class_for_worker(self, worker_id: int, num_worker: int, seed: int):
        """
        Recursively calls init_class_for_worker() on all properties of type MultiProcessSupport.
        Before calling _init_class_for_worker() on itself.

        Attention: We'll fire you if you override this method.
        this is the only solution in python that maybe prevent this method to be overridden,
        see https://stackoverflow.com/questions/2425656/how-to-prevent-a-function-from-being-overridden-in-python/2425785

        :param worker_id: worker id
        :param num_worker: number of workers
        :param seed: seed
        """
        members = inspect.getmembers(self)
        for name, obj in members:
            if issubclass(type(obj), MultiProcessSupport):
                obj.init_class_for_worker(worker_id, num_worker, seed)

        self._init_class_for_worker(worker_id, num_worker, seed)

    @abstractmethod
    def _init_class_for_worker(self, worker_id: int, num_worker: int, seed: int):
        pass


def mp_worker_init_fn(worker_id: int):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    if isinstance(dataset, MultiProcessSupport):
        dataset.init_class_for_worker(worker_info.id, worker_info.num_workers, worker_info.seed)
    else:
        print("dataset not an instance of MultiProcessSupport")
