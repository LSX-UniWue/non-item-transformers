local raw_dataset_path = "datasets/dataset/ml-1m/";
local cached_dataset_path = "/tmp/cache/ml-1m";
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
        cache_path: cached_dataset_path,
        // This template is equivalent to the explicit definition given in "data_sources"
        template: {
            name: "masked",
            split: "leave_percentage_out",
            file_prefix: dataset, // Optional
            num_workers: 4
        },
       /* data_sources: {
            split: "leave_one_out",
            file_prefix: dataset, // Optional
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
        },*/
        force_regeneration: "False",
        preprocessing: {
            extraction_directory: "/tmp/ml-1m/",
            output_directory: raw_dataset_path,
            min_item_feedback: 4,
            min_sequence_length: 4,
        }
    },
    templates: {
        unified_output: {
            path: output_path
        },
    },

    module: {
        type: "bert4rec",
        metrics: {
            full: {
                metrics: metrics
            },
        },
        model: {
            max_seq_length: max_seq_length,
            num_transformer_heads: 1,
            num_transformer_layers: 1,
            transformer_hidden_size: 32,
            transformer_dropout: 0.1,
            project_layer_type: 'linear'
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
                }
            }
        }
    },

    trainer: {
        loggers: {
            tensorboard: {}
        },
        checkpoint: {
            monitor: "recall@10",
            save_top_k: 3,
            mode: 'max'
        },
        gpus: 0,
        max_epochs: 10,
        check_val_every_n_epoch: 1
    }
}
