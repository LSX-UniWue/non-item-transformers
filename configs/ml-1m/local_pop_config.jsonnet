local raw_dataset_path = "datasets/dataset/ml-1m/";
local max_seq_length = 200;
local dataset = 'ml-1m';

local metrics =  {
    mrr: [1, 3, 5],
    recall: [1, 3, 5],
    ndcg: [1, 3, 5]
};
{
    templates: {
        unified_output: {
            path: "/tmp/experiments/pop"
        },
     },
     datamodule: {
        dataset: dataset,
        data_sources: {
            split: "ratio_split",
            file_prefix: dataset,
            num_workers: 4,
            train: {
                type: "session",
                processors: []
            },
            validation: {
                type: "next_item",
                processors: [
                    {
                        "type": "target_extractor"
                    },
                ]
            },
            test: {
             type: "next_item",
                processors: [
                    {
                        "type": "target_extractor"
                    },
                ]
            }
        },

        preprocessing: {
            extraction_directory: "/tmp/ml-1m/",
            output_directory: raw_dataset_path,
            min_item_feedback: 4,
            min_sequence_length: 4,
        }
    },
    module: {
        type: "pop",
        metrics: {
            full: {
                metrics: metrics
            },
            sampled: {
                sample_probability_file: "ml-1m.popularity.title.txt",
                num_negative_samples: 2,
                metrics: metrics
            }
        },
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
