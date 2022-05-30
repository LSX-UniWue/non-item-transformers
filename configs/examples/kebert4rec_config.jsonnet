local base_path = "../tests/example_dataset/";
local output_path = '/tmp/experiments/bert4rec';
local max_seq_length = 7;
local dataset = 'example';
local metrics =  {
    mrr: [1, 3, 5],
    recall: [1, 3, 5],
    ndcg: [1, 3, 5],
    rank: []
};
local tokenizer =  {
    special_tokens: {
      pad_token: "<PAD>",
      mask_token: "<MASK>",
      unk_token: "<UNK>"
        },
    };
{
    datamodule: {
        dataset: dataset,
        template: {
            name: "masked",
            split: "leave_one_out",
            file_prefix: dataset,
            num_workers: 0,
            batch_size: 2,
            mask_probability: 0.1,
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
        type: "kebert4rec",
        metrics: {
            full: {
                metrics: metrics
            },
            sampled: {
                sample_probability_file:  dataset + ".popularity.item_id.txt",
                num_negative_samples: 2,
                metrics: metrics
            },
            random_negative_sampled: {
                num_negative_samples: 2,
                metrics: metrics
            },
        },
        model: {
            max_seq_length: max_seq_length,
            num_transformer_heads: 1,
            num_transformer_layers: 1,
            transformer_hidden_size: 2,
            transformer_dropout: 0.1,
            additional_attributes: {
                attr_one: {
                    embedding_type: 'content_embedding'
                }
            }
        }
    },
    features: {
        item: {
            column_name: "item_id",
            sequence_length: max_seq_length,
            tokenizer: tokenizer
        },
        attr_one: {
            column_name: "attr_one",
            sequence_length: max_seq_length,
            tokenizer: tokenizer
        }
    },
    trainer: {
        loggers: {
            tensorboard: {},
            csv: {}
        },
        checkpoint: {
            monitor: "MRR@5",
            save_top_k: 3,
            mode: 'max'
        },
        early_stopping: {
          monitor: 'MRR@5',
          min_delta: 0.00,
          patience: 10,
          mode: 'max'
        },
        max_epochs: 5
    }
}
