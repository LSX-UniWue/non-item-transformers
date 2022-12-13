local raw_dataset_path = "../example_data/";
local dataset_path = "/Users/lisa/recommender/example_data/";
local base_path = "/Users/lisa/recommender/example_data/ratio_split-0.8_0.1_0.1/";
local max_seq_length = 7;
local prefix = 'example';
local dataset = 'example';
local output_path = '/Users/lisa/recommender/tmp/sasrec-output';
local metrics =  {
    mrr: [1, 3, 5],
    recall: [1, 3, 5],
    ndcg: [1, 3, 5],
    rank: []
};
{
    datamodule: {
        dataset: dataset,
            data_sources: {
                split: "ratio_split",
                file_prefix: dataset,
                train: {
                    type: "session",
                    processors: [
                        {"type": "target_extractor",
                        "parallel": "True",
                        },

                    ],
                    batch_size: 2
                },
                validation: {
                    type: "next_item",
                    processors: [
                        {"type": "target_extractor"}
                    ],
                     batch_size: 2
                },
                test: {
                 type: "next_item",
                    processors: [
                        {"type": "target_extractor"},
                    ],
                 batch_size: 2
                }
            },
        #template: {
        #    name: 'par_seq',
        #    split: 'ratio_split',
        #    file_prefix: dataset,
        #    batch_size: 2,
        #},
        preprocessing: {
            output_directory: dataset_path,
            input_file_path: "/Users/lisa/recommender/example/example.csv",
            ratio_split_min_sequence_length: 3,

        }
    },
    templates: {
        unified_output: {
            path: output_path
        },
    },
    module: {
        type: 'user-sasrec-full',
        metrics: {
            full: {
                metrics: metrics
            },
            sampled: {
                sample_probability_file: base_path + dataset +".popularity.item_id.txt",
                num_negative_samples: 2,
                metrics: metrics
            },
        },
        model: {
            transformer_hidden_size: 4,
            num_transformer_heads: 2,
            num_transformer_layers: 1,
            max_seq_length: max_seq_length,
            transformer_dropout: 0.1,
            segment_embedding: false,
            mode: "full",
            user_attributes: {
                user_id: {
                    embedding_type: 'user_linear_upscale'
                }
            }
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
            }
        },
        user_id: {
            type: "list",
            element_type: "str",
            delimiter: "|",
            sequence_length: max_seq_length,
            max_sequence_step_length: 2,
            column_name: "user_id",
            tokenizer: {
                special_tokens: {
                    pad_token: "<PAD>",
                    mask_token: "<MASK>",
                    unk_token: "<UNK>"
                },
                            vocabulary: {
                                file: base_path + "example.vocabulary.user_id-splitted.txt"
                            }
            }
        }
    },
    trainer: {
        loggers: {
            tensorboard: {}
        },
        checkpoint: {
            monitor: "MRR@5",
            save_top_k: 3,
            mode: 'max'
        },
        early_stopping: {
          monitor: 'recall@5',
          min_delta: 0.00,
          patience: 10,
          mode: 'max'
        },
        max_epochs: 3

    }
}