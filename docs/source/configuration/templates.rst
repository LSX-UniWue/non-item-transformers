.. _config templates:
Templates
======================================

.. code:: json
    templates: {
        unified_output: {
            path: output_path
        }
    },




TODO: Everything below was moved to datamodule

Parser Config
-------------

Common Constructs
-----------------

Here we list common data sources configurations.

Positional Datasource
~~~~~~~~~~~~~~~~~~~~~

TODO

Positive Negative Datasource
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This datasource returns the session excluding the last item as sequence
(key: ``TODO``) together with the successor for each sequence step
(positive example; key: ``TODO``), and a negative sampled item from the
item space, that does not occur in the session or is the successor.

.. code-block:: json

    ...
    {
        type: 'session',
        csv_file: '../tests/example_dataset/train.csv',
        csv_file_index: '../tests/example_dataset/train.idx',
        parser: {
            'item_column_name': 'column_name'
        },
        processors: [
            {
                type: 'tokenizer'
            },
            {
                type: 'pos_neg',
                'seed': 42
            }
        ]
    }
    ...

Templates for Specific Models
-----------------------------

Positive Negative DataSources Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This template adds data sources for

-  train (Positive Negative Datasource)
-  test (Positional Datasource)
-  validation (Positional Datasource)

The template is for models that use the complete sequence and train to
predict the successor for each sequence step and compare the scores for
the successor with a negative sample.

It can be triggered by adding the following element instead of
``data_sources``:

.. code-block:: json

    ...
    pos_neg_data_sources: {
        parser: {
            item_column_name: "column_name"
        },
        batch_size: BATCH_SIZE,
        max_seq_length: SEQ_LENGTH,
        path: "/path",
        train_file_prefix: "train"
        validation_file_prefix: "train",
        test_file_prefix: "train",
        seed: 42
    },
    ...

By default, the template configures the framework to

The following config parameters are available:

-  ``parser``: configs the parser for the csv file, see parser
   configuration
-  ``batch_size``: the batch size to use, if you want to override this
   for training, validation or test your model, add a
   ``{train,validation,test}_batch_size`` element to the element
-  ``seed``: the seed used to generate negative samples

Mask Datasource
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

    ...
    mask_data_sources: {
        loader: {
            batch_size: 9,
            num_workers: 0
        },
        path: base_path,
        file_prefix: prefix,
        mask_probability: 0.1,
        mask_seed: 123456,
        split_type: 'leave_one_out'
    }
    ...
