local base_path = '../tests/example_dataset/';
local output_path = "/tmp/experiments/sasrec_basket";
local max_seq_length = 7;
local dataset = 'example';
local metrics =  {
    recall: [1, 3, 5],
    ndcg: [1, 3, 5],
    f1: [1, 3, 5]
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
            batch_size: 9,
            seed: 123456
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
        type: "sasrec",
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
            transformer_hidden_size: 4,
            num_transformer_heads: 2,
            num_transformer_layers: 1,
            max_seq_length: max_seq_length,
            transformer_dropout: 0.1,
            embedding_pooling_type: 'mean'
        }
    },
    features: {
        item: {
            column_name: "item_id",
            type: "strlist",
            delimiter: ' + ',
            sequence_length: max_seq_length,
            max_sequence_step_length: 5,
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