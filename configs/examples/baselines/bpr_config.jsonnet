local base_path = "/mnt/c/Users/seife/work/recommender/tests/example_dataset/";
local output_path = '/tmp/experiments/bpr';
local max_seq_length = 7;
local dataset = 'example';
local metrics =  {
    mrr: [1, 3, 5],
    recall: [1, 3, 5],
    ndcg: [1, 3, 5]
};
{
    datamodule: {
        dataset: dataset,
        template: {
            name: "pos_neg",
            split: "ratio_split",
            file_prefix: dataset,
            num_workers: 0,
            batch_size: 9,
        },
        preprocessing: {
        }
    },
    templates: {
        unified_output: {
            path: output_path
        },
    },
    module: {
        type: "bpr",
        num_users: 1000,
        embedding_size: 16,
        regularization_factor: 0.1,
        metrics: {
            full: {
                metrics: metrics
            },
            sampled: {
                sample_probability_file: base_path + dataset + ".popularity.item_id.txt",
                num_negative_samples: 2,
                metrics: metrics
            },
            fixed: {
                item_file: base_path + dataset + ".relevant_items.item_id.txt",
                metrics: metrics
            }
        },
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
                }
            }
        },
        user: {
            column_name: "user_id",
            sequence: false,
            tokenizer: {
                special_tokens: {
                },
                vocabulary: {
                }
            }
        }
    },
    trainer: {
        max_epochs: 1,
        num_sanity_val_steps: 0,
        loggers: {
            tensorboard: {}
        },
        checkpoint: {
            monitor: "recall@5",
            save_top_k: 3,
            mode: 'max'
        },
    }
}
