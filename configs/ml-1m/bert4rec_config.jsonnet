local base_path = "../ml-1m/";
local output_path = "../ml-1m/bert4rec_d05/";
local hidden_size = 128;
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
        template: {
            name: "masked",
            split: "leave_one_out",
            path: base_path,
            file_prefix: dataset,
            num_workers: 4,
            batch_size: 64,
            max_seq_length: max_seq_length,
            mask_probability: 0.2,
            mask_seed: 42
        },
        preprocessing: {
            extraction_directory: "../ml-1m/",
            output_directory: base_path
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
                sample_probability_file: "ml-1m.popularity.title.txt",
                num_negative_samples: 100,
                metrics: metrics
            }
        },
        model: {
            max_seq_length: max_seq_length,
            num_transformer_heads: 2,
            num_transformer_layers: 2,
            transformer_hidden_size: hidden_size,
            transformer_dropout: 0.5
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
                    # Inferred by the datamodule
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
        gpus: 1,
        max_epochs: 800,
        check_val_every_n_epoch: 10
    }
}
