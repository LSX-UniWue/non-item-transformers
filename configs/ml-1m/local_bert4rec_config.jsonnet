local raw_dataset_path = "../datasets/dataset/ml-1m/";
local cached_dataset_path = raw_dataset_path;
local loo_path = cached_dataset_path + "loo/";
local output_path = "../dataset/ml-1m/exp/";
local max_seq_length = 200;
local metrics =  {
    mrr: [1, 5, 10],
    recall: [1, 5, 10],
    ndcg: [1, 5, 10]
};

local dataset = 'ml-1m';

{
    datamodule: {
        dataset: dataset,
        // This template is equivalent to the explicit definition given in "data_sources"
        /*template: {
            name: "masked",
            split: "leave_one_out",
            path: raw_dataset_path,
            file_prefix: dataset,
            num_workers: 4
        },*/
        data_sources: {
            split: "leave_one_out",
            path: raw_dataset_path,
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
            extraction_directory: "/tmp/ml-1m/",
            output_directory: raw_dataset_path,
            min_item_feedback: 0,
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
                sample_probability_file: loo_path + "ml-1m.popularity.title.txt",
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
            column_name: "title",
            sequence_length: max_seq_length,
            tokenizer: {
                special_tokens: {
                    pad_token: "<PAD>",
                    mask_token: "<MASK>",
                    unk_token: "<UNK>"
                },
                vocabulary: {
                    file: loo_path + "ml-1m.vocabulary.title.txt"
                }
            }
        }
    },

    trainer: {
        loggers: {
            tensorboard: {}
        },
        checkpoint: {
            monitor: "recall@10_sampled(100)",
            save_top_k: 3,
            mode: 'max'
        },
        gpus: 0,
        max_epochs: 10,
        check_val_every_n_epoch: 50
    }
}
