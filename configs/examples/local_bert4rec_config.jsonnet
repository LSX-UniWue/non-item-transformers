local base_path = '/Users/lisa/recommender/example/ratio_split-0.7_0.1_0.2/';
local dataset_path = "/Users/lisa/recommender/example/ratio_split-0.7_0.1_0.2/";
local output_path = '/Users/lisa/recommender/tmp/';
local ratio_path = '/Users/lisa/recommender/example/ratio_split-0.7_0.1_0.2/';
local max_seq_length = 7;
local dataset = 'example';
local metrics =  {
    mrr: [1, 3, 5],
    recall: [1, 3, 5],
    ndcg: [1, 3, 5],
    rank: []
};
{
    datamodule: {
        dataset: dataset,
        data_sources: {
            batch_size: 4,
            split: "ratio_split",
            path: dataset_path,
            file_prefix: dataset,
            train: {
                type: "session",
                processors: [
                    {
                        "type": "cloze",
                        "mask_probability": 0.2,
                        "only_last_item_mask_prob": 0.1
                    }
                ]
            },
            validation: {
                type: "session",
                processors: [
                    {
                        "type": "target_extractor"
                    },
                    {
                        "type": "last_item_mask"
                    }
                ]
            },
            test: {
             type: "session",
                processors: [
                    {
                        "type": "target_extractor"
                    },
                    {
                        "type": "last_item_mask"
                    }
                ]
            }
        },
        preprocessing: {
            input_file_path: "/Users/lisa/recommender/example_data/example.csv",
        }
    },
    templates: {
        unified_output: {
            path: output_path
        }
    },
    module: {
        type: "bert4rec",
        metrics: {
            full: {
                metrics: metrics
            },
            sampled: {
                sample_probability_file: ratio_path +"example.popularity.item_id.txt",
                num_negative_samples: 2,
                metrics: metrics
            },
            random_negative_sampled: {
                num_negative_samples: 2,
                metrics: metrics
            },
        },
        model: {
            max_seq_length: max_seq_length,
            num_transformer_heads: 1,
            num_transformer_layers: 1,
            transformer_hidden_size: 2,
            transformer_dropout: 0.1
        }
    },
    features: {
        item: {
            column_name: "item_id",
            sequence_length: max_seq_length,
            tokenizer: {
                special_tokens: {
                    pad_token: "<PAD>",
                    mask_token: "<MASK>",
                    unk_token: "<UNK>"
                },
            }
        }
    },
    trainer: {
        loggers: {
            tensorboard: {},
            csv: {}
        },
        checkpoint: {
            monitor: "recall@5",
            save_top_k: 3,
            mode: 'max'
        },
        early_stopping: {
          monitor: 'recall@5',
          min_delta: 0.00,
          patience: 10,
          mode: 'max'
        },
        max_epochs: 5
    },
           evaluation: {
                evaluators: [
                    {type: "sid", use_session_id: false},
                    {type: "recommendation"},
                    {type: "metrics"},
                   # {type: "input"},
                    #{type: "scores"},
                    {type: "target"},
                    ],
               # filter_items: {
                #    file: "/Users/lisa/recommender/configs/selected_items.csv"}, # compute metrics, recommendation and scores only for selected items
                number_predictions: 5,
                writer: {
                    type: "csv-single-line"
                }
                }
}
