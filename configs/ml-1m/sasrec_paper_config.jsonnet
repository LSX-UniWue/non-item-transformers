local base_path = "/ssd/ml-1m/";
local output_path = "/scratch/jane-doe-framework/experiments/ml-1m/sasrec_paper";
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
            name: "pos_neg",
            split: "leave_one_out",
            path: base_path,
            file_prefix: dataset,
            num_workers: 10,
            batch_size: 64
        },
        preprocessing: {
            input_directory: base_path,
            output_directory: base_path,
        }
    },
    templates: {
        unified_output: {
            path: output_path
        },
    },
    module: {
        type: "sasrec",
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
            transformer_hidden_size: 50,
            num_transformer_heads: 1,
            num_transformer_layers: 2,
            max_seq_length: max_seq_length,
            transformer_dropout: 0.2
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