#local raw_dataset_path = "datasets/dataset/ml-1m/";
local raw_dataset_path = "/tmp/datasets/ml-1m/";
local cached_dataset_path = raw_dataset_path;
local loo_path = cached_dataset_path + "loo/";
#local output_path = "../dataset/ml-1m/exp/";
local output_path = "/tmp/ml-1m/bert4rec/";
local max_seq_length = 200;
local metrics =  {
    mrr: [1, 5, 10],
    recall: [1, 5, 10],
    ndcg: [1, 5, 10]
};
local dataset = "ml-1m";
{
    datamodule: {
        dataset: dataset,
        template: {
            name: "masked",
            split: "leave_one_out",
            path: raw_dataset_path,
            file_prefix: dataset,
            num_workers: 4,
            batch_size: 64
        },
        force_regeneration: "False",
        preprocessing: {
            extraction_directory: raw_dataset_path + "raw/",
            output_directory: raw_dataset_path,
            min_item_feedback: 5,
            min_sequence_length: 2,
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
            }
        },
        model: {
            max_seq_length: max_seq_length,
            num_transformer_heads: 2,
            num_transformer_layers: 2,
            transformer_hidden_size: 8,
            transformer_dropout: 0.1,
            nonlinearity: "relu"
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
                }
            }
        }
    },
    trainer: {
        loggers: {
            tensorboard: {}
        },
        checkpoint: {
            monitor: "recall@10",
            save_top_k: 3,
            mode: 'max'
        },
        gpus: 1,
        max_epochs: 10,
        check_val_every_n_epoch: 1,
        limit_train_batches: 512
    }
}
