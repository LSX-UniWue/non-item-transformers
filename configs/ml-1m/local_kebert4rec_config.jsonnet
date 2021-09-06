local base_path = "datasets/dataset/ml-1m/";
local output_path = "dataset/datassts/ml-1m/exp/";
local max_seq_length = 10;
local metrics =  {
    mrr: [1, 5, 10],
    recall: [1, 5, 10],
    ndcg: [1, 5, 10]
};

local dataset = 'ml-1m';

local additional_attributes_list = ['genres'];

{
    datamodule: {
        dataset: dataset,
        template: {
            name: "masked",
            split: "leave_one_out",
            path: base_path,
            file_prefix: dataset,
            num_workers: 0,
            batch_size: 4,
            mask_probability: 0.2,
            mask_seed: 42,
            mask_additional_attributes: additional_attributes_list
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
        type: "kebert4rec",
        additional_attributes: additional_attributes_list,
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
            max_seq_step_length: 5,
            num_transformer_heads: 1,
            num_transformer_layers: 1,
            transformer_hidden_size: 32,
            transformer_dropout: 0.2,
            project_layer_type: 'linear',
            additional_attributes: {
                genres: {
                    embedding_type: 'linear_upscale',
                    vocab_size: 301
                }
            }
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
        },
        genres: {
            type: "strlist",
            delimiter: "|",
            sequence_length: max_seq_length,
            max_sequence_step_length: 5,
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
        },
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
        max_epochs: 10,
        check_val_every_n_epoch: 50
    }
}
