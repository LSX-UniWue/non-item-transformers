.. _config module:
Module
======

Metrics
-------

The following metrics are implemented in this framework with the keys to
use for configuration:

-  Recall / HR (``recall``)
-  Precision (``precision``)
-  F1 (``F1``)
-  DCG (``DCG``)
-  NDCG (``NDCG``)
-  MRR (``MRR``)

For each metric you can provide one or more different ``k``\ s to
evaluate the Metric@k value. The metrics can be access (e.g. in the
checkpoint), via ``KEY@k``.

- MRR full (``mrr_full``)
- Rank (``rank``)

TODO: Metrics will be covered by TorchMetrics soon.


Metric Evaluation
-----------------

There are three evaluation strategies available in the framework:

-  ``full``: the metrics are evaluated on the complete item space
-  ``sampled``: the metrics are evaluated on the positive item(s) and
   ``s`` sampled negative items (given a probability)
-  ``fixed``: the metrics are evaluated on a fixed subset of the item
   space

Full
~~~~

If you want to validate your model or evaluate your model how good your
model ranks all items of the item space, you can specify a metrics
section under module. For each metric you can specify which ``k``\ s
should be evaluated.

.. code:: json

    ...
    module: {
            ...
            metrics: {
                full: {
                    metrics: {
                        recall: [1, 3, 5]
                    }
                },
                ...
            },
            ...
    }

Sampled Metrics
~~~~~~~~~~~~~~~

In contrast to metrics the sampled metrics configuration only samples
items from the item space to evaluate it with target item(s).

.. code:: json

    ...
    module: {
            ...
            metrics: {
                sampled: {
                    sample_probability_file: PATH_TO_FILE,
                    num_negative_samples: 100,
                    metrics: {
                        recall: [1, 3, 5]
                    }
                },
                ...
            },
            ...
    }

The ``sampled`` metrics config requires the following parameters:

-  ``sample_probability_file``: The configurable file contains in the
   i-th line the probability of the (i-1) item based on the vocabulary
   files.
-  ``num_negative_samples``: The number of negative samples to draw from
   the provided probability file.
-  ``metrics``: you can define all metrics you can also define using all
   items of the dataset.

The probability file is generated automatically during the dataset generation process.

Fixed Subset
~~~~~~~~~~~~

This metric only evaluates a fixed set of items.

.. code:: json

    ...
    module: {
            ...
            metrics: {
                fixed: {
                    item_file: PATH_TO_FILE,
                    metrics: {
                        recall: [1, 3, 5]
                    }
                },
                ...
            },
            ...
    }

The ``fixed`` metrics config requires the following parameters:

-  ``item_file``: The configurable file contains the item ids of the
   subset to evaluate (item id line by line).
-  ``metrics`` you can define all metrics you can also define using all
   items of the dataset.


Models
--------

.. code::json

    ...
    module: {
        type: MODEL_NAME,
        model: {
            max_seq_length: max_seq_length,
            <parameter 1>: <value>,
            <parameter 2>: <value>,
            ...
        },
        ...
    },
    ...


The following models and corresponding parameters are implemented:

+--------------+--------------------------------------------+
| Model Name   | Model Parameter                            |
+==============+============================================+
| bert4rec     | transformer_hidden_size,                   |
|              | num_transformer_heads,                     |
|              | num_transformer_layers,                    |
|              | item_vocab_size,                           |
|              | max_seq_length,                            |
|              | transformer_dropout,                       |
|              | project_layer_type (optional),             |
|              | embedding_pooling_type (optional),         |
|              | initializer_range (optional),              |
|              | transformer_intermediate_size (optional),  |
|              | transformer_attention_dropout (optional)   |
+--------------+--------------------------------------------+
| kebert4rec   | transformer_hidden_size,                   |
|              | num_transformer_heads,                     |
|              | num_transformer_layers,                    |
|              | item_vocab_size,                           |
|              | max_seq_length,                            |
|              | transformer_dropout,                       |
|              | additional_attributes,                     |
|              | embedding_pooling_type (optional),         |
|              | initializer_range (optional),              |
|              | transformer_intermediate_size (optional),  |
|              | transformer_attention_dropout (optional)   |
+--------------+--------------------------------------------+
| caser        | embedding_size,                            |
|              | item_vocab_size,                           |
|              | user_vocab_size,                           |
|              | max_seq_length,                            |
|              | num_vertical_filters,                      |
|              | num_horizontal_filters,                    |
|              | conv_activation_fn,                        |
|              | fc_activation_fn,                          |
|              | dropout,                                   |
|              | embedding_pooling_type (optional)          |
+--------------+--------------------------------------------+
| narm         | item_vocab_size,                           |
|              | item_embedding_size,                       |
|              | global_encoder_size,                       |
|              | global_encoder_num_layers,                 |
|              | embedding_dropout,                         |
|              | context_dropout,                           |
|              | batch_first (optional),                    |
|              | embedding_pooling_type (optional)          |
+--------------+--------------------------------------------+
| sasrec       | transformer_hidden_size,                   |
|              | num_transformer_heads,                     |
|              | num_transformer_layers,                    |
|              | item_vocab_size,                           |
|              | max_seq_length,                            |
|              | transformer_dropout,                       |
|              | embedding_pooling_type (optional),         |
|              | transformer_intermediate_size (optional),  |
|              | transformer_attention_dropout (optional)   |
+--------------+--------------------------------------------+
| rnn /        | cell_type,                                 |
| dream        | item_vocab_size,                           |
|              | item_embedding_dim,                        |
|              | hidden_size,                               |
|              | num_layers,                                |
|              | dropout,                                   |
|              | bidirectional (optional),                  |
|              | nonlinearity (optional),                   |
|              | embedding_pooling_type (optional),         |
|              | project_layer_type (optional)              |
+--------------+--------------------------------------------+
| cosrec       | user_vocab_size,                           |
|              | item_vocab_size,                           |
|              | embed_dim,                                 |
|              | block_num,                                 |
|              | block_dim,                                 |
|              | fc_dim,                                    |
|              | activation_function,                       |
|              | dropout,                                   |
|              | embedding_pooling_type (optional)          |
+--------------+--------------------------------------------+
| hgn          | user_vocab_size,                           |
|              | item_vocab_size,                           |
|              | num_successive_items,                      |
|              | dims,                                      |
|              | embedding_pooling_type (optional)          |
+--------------+--------------------------------------------+
| nnrec        | item_vocab_size,                           |
|              | user_vocab_size,                           |
|              | item_embedding_size,                       |
|              | user_embedding_size,                       |
|              | hidden_size,                               |
|              | max_sequence_length,                       |
|              | embedding_pooling_type                     |
+--------------+--------------------------------------------+

Additionally, the following baselines are implemented:

    *  bpr
    *  pop
    *  session_pop
    *  markov


