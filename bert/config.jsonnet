{
    "datamodule": {
        "data_sources": {
            "file_prefix": "example",
            "path": "/Users/lisa/recommender/example/ratio_split-0.8_0.1_0.1/",
            "split": "ratio_split",
            "test": {
                "processors": [
                    {
                        "type": "target_extractor"
                    },
                    {
                        "type": "last_item_mask"
                    }
                ],
                "type": "session"
            },
            "train": {
                "processors": [
                    {
                        "mask_probability": 0.2,
                        "only_last_item_mask_prob": 0.1,
                        "type": "cloze"
                    }
                ],
                "type": "session"
            },
            "validation": {
                "processors": [
                    {
                        "type": "target_extractor"
                    },
                    {
                        "type": "last_item_mask"
                    }
                ],
                "type": "session"
            }
        },
        "dataset": "example",
        "preprocessing": {
            "min_sequence_length": 2
        }
    },
    "features": {
        "item": {
            "column_name": "item_id",
            "sequence_length": 7,
            "tokenizer": {
                "special_tokens": {
                    "mask_token": "<MASK>",
                    "pad_token": "<PAD>",
                    "unk_token": "<UNK>"
                },
                "vocabulary": {
                    "file": "/Users/lisa/recommender/example/ratio_split-0.8_0.1_0.1/example.vocabulary.item_id.txt"
                }
            }
        }
    },
    "module": {
        "metrics": {
            "full": {
                "metrics": {
                    "mrr": [
                        1,
                        3,
                        5
                    ],
                    "ndcg": [
                        1,
                        3,
                        5
                    ],
                    "rank": [],
                    "recall": [
                        1,
                        3,
                        5
                    ]
                }
            },
            "random_negative_sampled": {
                "metrics": {
                    "mrr": [
                        1,
                        3,
                        5
                    ],
                    "ndcg": [
                        1,
                        3,
                        5
                    ],
                    "rank": [],
                    "recall": [
                        1,
                        3,
                        5
                    ]
                },
                "num_negative_samples": 2
            },
            "sampled": {
                "metrics": {
                    "mrr": [
                        1,
                        3,
                        5
                    ],
                    "ndcg": [
                        1,
                        3,
                        5
                    ],
                    "rank": [],
                    "recall": [
                        1,
                        3,
                        5
                    ]
                },
                "num_negative_samples": 2,
                "sample_probability_file": "/Users/lisa/recommender/example/ratio_split-0.8_0.1_0.1/example.popularity.item_id.txt"
            }
        },
        "model": {
            "max_seq_length": 7,
            "num_transformer_heads": 1,
            "num_transformer_layers": 1,
            "transformer_dropout": 0.1,
            "transformer_hidden_size": 2
        },
        "type": "bert4rec"
    },
    "trainer": {
        "checkpoint": {
            "mode": "max",
            "monitor": "recall@5",
            "save_top_k": 3,
            "dirpath": "/Users/lisa/recommender/tmp//checkpoints",
            "filename": "{epoch}-{recall@5}",
            "save_last": true
        },
        "early_stopping": {
            "min_delta": 0,
            "mode": "max",
            "monitor": "recall@5",
            "patience": 10
        },
        "loggers": {
            "csv": {
                "save_dir": "/Users/lisa/recommender/tmp/",
                "name": "",
                "version": ""
            },
            "tensorboard": {
                "save_dir": "/Users/lisa/recommender/tmp/",
                "name": "",
                "version": ""
            }
        },
        "max_epochs": 5
    }
}