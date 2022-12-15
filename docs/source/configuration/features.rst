.. _config features:
Features
=========

This config part specifies the feature(s) considered during training. If the model supports it, multiple items can be listed successively.

* *column_name*: column name of the item in the dataset (e.g. `title`).
* *sequence_length*: maximum length of a sequence
* *tokenizer*:
    * *special_tokens*: dictionary containing the special tokens that are to be included in the vocabulary
    * *vocabulary*: path to the vocabulary file. Accepted values include an absolute path or a relative path within the dataset path. The value can be left out if the vocabulary file is created during preprocessing.


Example
~~~~~~~~

.. code:: json
    {
        ...
        features: {
            item: {
                column_name: "title",
                sequence_length: 200,
                tokenizer: {
                    special_tokens: {
                        pad_token: "<PAD>",
                        mask_token: "<MASK>",
                        unk_token: "<UNK>"
                    },
                    vocabulary: {
                        file: "ml-1m.popularity.title.txt"
                    }
                }
            }
        },
        ...


