from asme.datasets.dataset_index_splits.ratio_split_strategy import RatioSplitStrategy


def get_ratio_strategy(train_ratio: float,
                       test_ratio: float,
                       validation_ratio: float,
                       seed: int
                       ) -> RatioSplitStrategy:
    return RatioSplitStrategy(train_ratio, test_ratio, validation_ratio, seed)
