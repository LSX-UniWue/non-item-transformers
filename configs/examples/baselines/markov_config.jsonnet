local base_path = "../tests/example_dataset/";
local output_path = '/tmp/experiments/markov';
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
            name: "next_sequence_step",
            split: "ratio_split",
            file_prefix: dataset,
            num_workers: 0
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
        type: "markov",
        metrics: {
            full: {
                metrics: metrics
            },
            sampled: {
                sample_probability_file: base_path + dataset + ".popularity.item_id.txt",
                num_negative_samples: 2,
                metrics: metrics
            },
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
        }
    },
    trainer: {
        max_epochs: 1,
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
