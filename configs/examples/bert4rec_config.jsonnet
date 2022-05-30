local base_path = '../tests/example_dataset/';
local output_path = '/tmp/experiments/sasrec';
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
        cache_path: "/tmp/cache",
        dataset: dataset,
        /*template: {
            name: "masked",
            split: "leave_one_out",
            path: dataset_path,
            file_prefix: dataset,
            num_workers: 0
        },*/
        data_sources: {
            split: "ratio_split",
            #path: dataset_path,
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
            min_sequence_length: 2,
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
                sample_probability_file: base_path + dataset + ".popularity.item_id.txt",
                num_negative_samples: 2,
                metrics: metrics
            },
            random_negative_sampled: {
                num_negative_samples: 2,
                metrics: metrics
            },
            #fixed: {
            #    item_file: dataset_path + "loo/example.relevant_items.item_id.txt",
            #    metrics: metrics
            #}
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
                vocabulary: {
                    #file: "example.vocabulary.item_id.txt"
                }
            }
        },
         session_identifier: {
                    column_name: "session_id",
                    sequence_length: max_seq_length,
                    sequence: false,
                    run_tokenization: false,
         },
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
            {type: "sid", use_session_id: true},
            {type: "recommendation"},
        #    {type: "metrics"},
        #    {type: "input"},
        #    {type: "scores"},
        #    {type: "target"},
            ],
        #selected_items_file: "/Users/lisa/recommender/configs/selected_items.csv",
        number_predictions: 5
        }
}
