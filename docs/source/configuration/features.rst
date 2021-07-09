.. _config features:
Features
=========

'type', 'sequence', 'column_name', "tokenizer"

.. code:: json
    {
        ...
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
        ...