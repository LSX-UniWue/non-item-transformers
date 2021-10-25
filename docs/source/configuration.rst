.. _main_configuration:

Training Configuration
======================================

The configuration for a training contains 5 main sections:

-  :ref:`Datamodule <config datamodule>`: contains information about the dataset
-  :ref:`Templates <config templates>`: contains the output directory
-  :ref:`Module <config module>`: contains the model, hyperparameter and metrics
-  :ref:`Features <config features>`: contains information about the feature(s) and the tokenization process
-  :ref:`Trainer <config trainer>`: contains information about the training itself


Example
"""""""""
The example below contains configurations for training Bert4Rec on the Movielens-1M data set.

.. code:: json

    local base_path = "/example/path/ml-1m/";
    local loo_path = base_path + "loo/";
    local output_path = "/example/output/path/";
    local hidden_size = 128;
    local max_seq_length = 200;
    local metrics =  {
        recall: [1, 5, 10],
        ndcg: [1, 5, 10]
    };
    local dataset = 'ml-1m';

    {
        datamodule: {
            dataset: dataset,
            template: {
                name: "masked",
                split: "leave_one_out",
                path: dataset_path,
                file_prefix: dataset,
                num_workers: 4,
                batch_size: 64,
                max_seq_length: max_seq_length,
                mask_probability: 0.2,
                mask_seed: 42
            },
            force_regeneration: "False",
            preprocessing: {
                extraction_directory: "/tmp/ml-1m/",
                output_directory: raw_dataset_path,
                min_item_feedback: 2,
                min_sequence_length: 2
            }
        },
        templates: {
            unified_output: {
                path: output_path
            }
        },
        module: {
            type: "bert4rec",
            metrics: {
                full: {
                    metrics: metrics
                },
                sampled: {
                  sample_probability_file: loo_path + dataset + ".popularity.title.txt",
                    num_negative_samples: 100,
                    metrics: metrics
                }
            },
            model: {
                max_seq_length: max_seq_length,
                num_transformer_heads: 2,
                num_transformer_layers: 2,
                transformer_hidden_size: hidden_size,
                transformer_dropout: 0.5
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
                        file: loo_path + dataset + ".vocabulary.title.txt"
                    }
                }
            }
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
            gpus: 1,
            max_epochs: 800,
            check_val_every_n_epoch: 10
        }
    }
