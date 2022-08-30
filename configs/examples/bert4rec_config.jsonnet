#Beispielconfig
local base_path = '../tests/example_dataset/';
local output_path = '/tmp/experiments/bert4rec';
local max_seq_length = 7;
local dataset = 'example';
local metrics =  {
    mrr: [1, 3, 5],
    recall: [1, 3, 5],
    ndcg: [1, 3, 5],
    rank: []
};
local tokenizer= {
    special_tokens: {
      pad_token: "<PAD>",
      mask_token: "<MASK>",
      unk_token: "<UNK>"}
    };
{
    datamodule: {
        cache_path: "/tmp/cache",
        dataset: dataset,
        data_sources: {
            split: "ratio_split",
            #path: dataset_path,
            file_prefix: dataset,
            train: {
                type: "session",
                processors: [
                    {
                        "type": "cloze",
                        "mask_probability": 0.2,
                        "only_last_item_mask_prob": 0.1
                    }
                ]
            },
            validation: {
                type: "session",
                processors: [
                    {
                        "type": "target_extractor"
                    },
                    {
                        "type": "last_item_mask"
                    }
                ]
            },
            test: {
             type: "session",
                processors: [
                    {
                        "type": "target_extractor"
                    },
                    {
                        "type": "last_item_mask"
                    }
                ]
            }
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
        type: "bert4rec",
        metrics: {
            full: {
                metrics: metrics
            },
        },
        model: {
            max_seq_length: max_seq_length,
            num_transformer_heads: 1,
            num_transformer_layers: 1,
            transformer_hidden_size: 2,
            transformer_dropout: 0.1
        }
    },
    features: {
        item: {
            column_name: "item_id",
            sequence_length: max_seq_length,
            tokenizer: tokenizer
            },
        session_identifier: {
                    column_name: "session_id",
                    sequence_length: max_seq_length,
                    sequence: false,
                    run_tokenization: false,
         },

        },
    trainer: {
        loggers: {
            tensorboard: {},
            csv: {}
        },
        checkpoint: {
            monitor: "recall@5",
            save_top_k: 3,
            mode: 'max'
        },
        early_stopping: {
          monitor: 'recall@5',
          min_delta: 0.00,
          patience: 10,
          mode: 'max'
        },
        max_epochs: 5
    },
    evaluation: {
            evaluators: [
                {type: "sid", use_session_id: true},
                {type: "recommendation"},
            #    {type: "metrics"},
            #    {type: "input"},
            #    {type: "scores"},
            #    {type: "target"},
                ],
            number_predictions: 30,
            writer: {type: "csv-single-line"
            }
    }
}
