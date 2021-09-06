local base_path = "/ssd/ml-20m/";
local output_path = "/scratch/jane-doe-framework/experiments/ml-20m/bert4rec_new0.1";
local max_seq_length = 200;
local metrics =  {
    mrr: [1, 5, 10],
    recall: [1, 5, 10],
    ndcg: [1, 5, 10]
};

local dataset = 'ml-20m';

{

    datamodule: {
        dataset: dataset,
        template: {
            name: "masked",
            split: "leave_one_out",
            path: base_path,
            file_prefix: dataset,
            batch_size: 64
            max_seq_length: max_seq_length,
            mask_probability: 0.2,
            mask_seed: 42
        },
        preprocessing: {
            extraction_directory: "/tmp/ml-1m/",
            output_directory: base_path,
            min_item_feedback: 4,
            min_sequence_length: 4,
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
                sample_probability_file: "ml-20m.popularity.title.txt",
                num_negative_samples: 100,
                metrics: metrics
            }
        },
        model: {
            max_seq_length: max_seq_length,
            num_transformer_heads: 8,
            num_transformer_layers: 2,
            transformer_hidden_size: 256,
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
        gpus: 8,
        max_epochs: 800,
        accelerator: "ddp",
        check_val_every_n_epoch: 100,
        gradient_clip_val: 5
    }
}
