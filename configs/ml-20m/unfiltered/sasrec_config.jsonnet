local base_path = "/ssd/ml-20m/";
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
            path: "/scratch/jane-doe-framework/experiments/ml-20m/sasrec_new256_8"
        }
    },
    module: {
        type: "sasrec",
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
            transformer_hidden_size: 256,
            num_transformer_heads: 8,
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
        gpus: 8,
        max_epochs: 800,
        accelerator: "ddp",
        check_val_every_n_epoch: 100
    }
}