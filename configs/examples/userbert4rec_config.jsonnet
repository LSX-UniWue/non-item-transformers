local raw_dataset_path = "../example_data/";
local dataset_path = "/Users/lisa/recommender/example_data/";
local base_path = "/Users/lisa/recommender/example_data/ratio_split-0.8_0.1_0.1/";
local max_seq_length = 7;
local prefix = 'example';
local dataset = 'example';
local output_path = '/Users/lisa/recommender/tmp/bert4rec-output';
local metrics =  {
    mrr: [1, 3, 5],
    recall: [1, 3, 5],
    ndcg: [1, 3, 5],
    rank: []
};
{
    datamodule: {
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

                ],
                batch_size: 2
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
                ],
                                 batch_size: 2
            },
            test: {
             type: "session",
                processors: [
                    {
                        "type": "no_target_extractor"
                    },
                    {
                        "type": "last_item_mask"
                    }
                ],
             batch_size: 2
            }
        },
        preprocessing: {
            output_directory: dataset_path,
            min_sequence_length: 2
        }
    },
    templates: {
        unified_output: {
            path: output_path
        }
    },
    module: {
        type: "ubert4rec",
        metrics: {
            full: {
                metrics: metrics
            },
            sampled: {
                sample_probability_file: base_path + "example.popularity.item_id.txt",
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
            num_transformer_layers: 3,
            transformer_hidden_size: 2,
            transformer_dropout: 0.1,
            segment_embedding: true,
            additional_attributes: {
                attr_one: {
                    embedding_type: 'content_embedding',
                    vocab_size: 7
                }
            },
            user_attributes: {
                user_one: {
                    embedding_type: 'user_embedding',
                    vocab_size: 5
                }
            }
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
                    file: base_path + "example.vocabulary.item_id.txt"
                }
            }
        },
        attr_one: {
            sequence_length: max_seq_length,
            tokenizer: {
                special_tokens: {
                    pad_token: "<PAD>",
                    mask_token: "<MASK>",
                    unk_token: "<UNK>"
                },
                vocabulary: {
                    file: base_path + "example.vocabulary.attr_one.txt"
                }
            }
        },
        user_one: {
            sequence: false,
            tokenizer: {
                special_tokens: {
                    pad_token: "<PAD>",
                    mask_token: "<MASK>",
                    unk_token: "<UNK>"
                },
                vocabulary: {
                    file: base_path + "example.vocabulary.user_one.txt"
                }
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
    }
}
