local base_path = "../tests/example_dataset/";
local output_path = '/tmp/experiments/hgn';
local max_seq_length = 7;
local dataset = 'example';
local num_successive_items = 3;
local metrics =  {
    mrr: [1, 3, 5],
    recall: [1, 3, 5],
    ndcg: [1, 3, 5]
};

{
    datamodule: {
        cache_path: "/tmp/ssd",
        dataset: dataset,
        template: {
            name: "pos_neg",
            split: "ratio_split",
            file_prefix: dataset,
            num_workers: 0,
            batch_size: 9
        },
        preprocessing: {
        }
    },
    templates: {
        unified_output: {
            path: "/tmp/experiments/hgn"
        }
    },
    module: {
        type: "hgn",
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
        model: {
            user_vocab_size: 0,
            dims: 50,
            num_successive_items: num_successive_items
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
                }
            }
        }
    },
    trainer: {
        loggers: {
            tensorboard: {}
        },
        checkpoint: {
            monitor: "recall@5",
            save_top_k: 3,
            mode: 'max'
        }
    }
}