local raw_dataset_path = "../tests/example_dataset/";
local cached_dataset_path = raw_dataset_path;
local dataset_path = "/tmp/example/";
local max_seq_length = 7;
local prefix = 'example';
local dataset = 'example';
local output_path = '/tmp/bert4rec-output';
local metrics =  {
    mrr: [1, 3, 5],
    recall: [1, 3, 5],
    ndcg: [1, 3, 5],
    rank: []
};
{
    datamodule: {
        dataset: dataset,
        template: {
            name: "masked",
            split: "leave_one_out",
            path: dataset_path,
            file_prefix: dataset,
            num_workers: 0
        },
        /*data_sources: {
            split: "leave_one_out",
            path: raw_dataset_path,
            file_prefix: dataset,
            train: {
                type: "session"
            },
            validation: {
                type: "session"
            },
            test: {
                type: "session"
            }
        },*/
        preprocessing: {
            output_directory: dataset_path,
            min_sequence_length: 2
        }
    },
    templates: {
        unified_output: {
            path: output_path
        },
        /*mask_data_sources: {
            parser: {
                item_column_name: "title"
            },
            loader: {
                batch_size: 4,
                max_seq_length: max_seq_length
            },
            path: cached_dataset_path,
            file_prefix: file_prefix,
            split_type: 'leave_one_out',
            mask_probability: 0.2,
            mask_seed: 42
        } */
    },
    module: {
        type: "bert4rec",
        metrics: {
            full: {
                metrics: metrics
            },
            sampled: {
                sample_probability_file: dataset_path + "loo/example.popularity.item_id.txt",
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
                    file: dataset_path + "loo/example.vocabulary.item_id.txt"
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
