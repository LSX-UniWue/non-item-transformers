local base_path = "../tests/example_dataset/";
local output_path = '/tmp/experiments/caser';
local number_of_targets = 1;
local markov_length = 2;
local max_seq_length = markov_length;
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
            name: "sliding_window",
            split: "ratio_split",
            file_prefix: dataset,
            num_workers: 0,
            batch_size: 9,
            pad_direction: "left",
            dynamic_padding: false,
            window_markov_length: markov_length,
            window_target_length: number_of_targets
        },
        preprocessing: {
        }
    },
    templates: {
        unified_output: {
            path: output_path
        }
    },
    module: {
        type: "caser",
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
            dropout: 0.1,
            max_seq_length: max_seq_length,
            embedding_size: 4,
            num_vertical_filters: 2,
            num_horizontal_filters: 1,
            fc_activation_fn: "relu",
            conv_activation_fn: "relu",
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