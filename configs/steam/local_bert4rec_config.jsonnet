local raw_dataset_path = "datasets/dataset/steam/";
local cached_dataset_path = raw_dataset_path;
local loo_path = cached_dataset_path + "loo/";
local output_path = "../dataset/steam/exp/";
local max_seq_length = 200;
local metrics =  {
    mrr: [1, 5, 10],
    recall: [1, 5, 10],
    ndcg: [1, 5, 10]
};

local dataset = 'steam';

{
    datamodule: {
        dataset: dataset,
        template: {
            name: "masked",
            split: "leave_one_out",
            path: raw_dataset_path,
            file_prefix: dataset,
            num_workers: 4
        },

        preprocessing: {
            input_dir: raw_dataset_path,
            output_directory: raw_dataset_path,
            min_item_feedback: 5,
            min_sequence_length: 5
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
            sampled: {
                sample_probability_file: "steam.popularity.product_id.txt",
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
            transformer_hidden_size: 32,
            transformer_dropout: 0.1,
            project_layer_type: 'linear'
        }
    },
    features: {
        item: {
            column_name: "product_id",
            sequence_length: max_seq_length,
            tokenizer: {
                special_tokens: {
                    pad_token: "<PAD>",
                    mask_token: "<MASK>",
                    unk_token: "<UNK>"
                },
                vocabulary: {
                    #file: loo_path + file_prefix + "vocabulary.title.txt"
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
