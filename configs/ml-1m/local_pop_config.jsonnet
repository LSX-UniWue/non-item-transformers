local raw_dataset_path = "./datasets/dataset/ml-1m/";
local output_path = "./datasets/ml-1m/output/";
local extr_dir = "./datasets/ml-1m-raw/";
local max_seq_length = 200;
local prefix = 'ml-1m';
local dataset = 'ml-1m';
local metrics =  {
    mrr: [1, 5, 10],
    recall: [1, 5, 10],
    ndcg: [1, 5, 10],
    rank: []
};
{
    templates: {
        unified_output: {
            path: "/Users/lisa/recommender/pop_ml/"
        },
     },
     datamodule: {
        dataset: dataset,
        data_sources: {
            split: "leave_one_out",
            file_prefix: dataset,
            num_workers: 4,
            train: {
                type: "session",
                processors: []
            },
            validation: {
                type: "session",
                processors: [
                    {
                        "type": "target_extractor"
                    },
                ]
            },
            test: {
             type: "session",
                processors: [
                    {
                        "type": "target_extractor"
                    },
                ]
            }
        },
        preprocessing: {
            extraction_directory: extr_dir,
            output_directory: raw_dataset_path,
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
