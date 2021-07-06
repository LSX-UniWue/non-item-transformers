local raw_dataset_path = "../datasets/dataset/dota-shop/";
local cached_dataset_path = "/tmp/ssd/";
local loo_path = cached_dataset_path + "loo/";
local output_path = "../dataset/dota-shop/exp/";
local max_seq_length = 200;
local metrics =  {
    mrr: [1, 5, 10],
    recall: [1, 5, 10],
    ndcg: [1, 5, 10]
};

local file_prefix = 'dota-shop';

{
    datamodule: {
        dataset: "dota-shop",
        cache_path: "/tmp/ssd",
        data_sources: {
            type: "mask"
        },
        preprocessing: {
            output_directory: raw_dataset_path,
            raw_csv_file_path: raw_dataset_path + "dota-shop-raw.csv",
            min_sequence_length: 2
        }
    },
    templates: {
        unified_output: {
            path: output_path
        },
        /*mask_data_sources: {
            parser: {
                item_column_name: "item_id"
            },
            loader: {
                batch_size: 4,
                max_seq_length: max_seq_length
            },
            path: cached_dataset_path,
            file_prefix: file_prefix,
            split_type: 'leave_one_out',
            mask_probability: 0.2,
            mask_seed: 42
        }*/
    },
    module: {
        type: "bert4rec",
        metrics: {
            full: {
                metrics: metrics
            },
            sampled: {
              sample_probability_file: loo_path + file_prefix + ".popularity.item_id.txt",
                num_negative_samples: 100,
                metrics: metrics
            }
        },
        model: {
            max_seq_length: max_seq_length,
            num_transformer_heads: 1,
            num_transformer_layers: 1,
            transformer_hidden_size: 32,
            transformer_dropout: 0.2,
            project_layer_type: 'linear'
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
                    file: loo_path + file_prefix+".vocabulary.item_id.txt"
                }
            }
        },
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
        gpus: 0,
        max_epochs: 10,
        check_val_every_n_epoch: 50
    }
}
