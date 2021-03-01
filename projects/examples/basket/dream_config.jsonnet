local base_path = "../tests/example_dataset/";
local max_seq_length = 7;
local metrics =  {
    recall: [1, 3, 5],
    ndcg: [1, 3, 5],
    f1: [1, 3, 5]
};
{
    templates: {
        unified_output: {
            path: "/tmp/experiments/dream_basket"
        },
        next_sequence_step_data_sources: {
            parser: {
                item_column_name: "item_id",
                item_separator: ' + '
            },
            loader: {
                batch_size: 9,
                max_seq_length: max_seq_length,
                max_seq_step_length: 5
            },
            path: base_path,
            validation_file_prefix: "train",
            test_file_prefix: "train"
        }
    },
    module: {
        type: "dream",
        metrics: {
            full: {
                metrics: metrics
            },
            sampled: {
                sample_probability_file: base_path + "popularity.txt",
                num_negative_samples: 2,
                metrics: metrics
            },
            fixed: {
                item_file: base_path + "relevant_items.txt",
                metrics: metrics
            }
        },
        model: {
            cell_type: "lstm",
            item_embedding_dim: 4,
            hidden_size: 4,
            num_layers: 1,
            dropout: 0.0,
            embedding_pooling_type: 'mean'
        }
    },
    tokenizers: {
            item: {
                tokenizer: {
                    special_tokens: {
                        pad_token: "<PAD>",
                        mask_token: "<MASK>",
                        unk_token: "<UNK>"
                    },
                    vocabulary: {
                        file: base_path + "vocab.txt"
                    }
                }
            }
        },
    trainer: {
        logger: {
            type: "tensorboard"
        },
        checkpoint: {
            monitor: "recall@5",
            save_top_k: 3,
            mode: 'max'
        }
    }
}