.. _User Guide:
User Guide
==========

Prerequisite:
- Built the virtual environment as described in the `Project Overview <./project_overview.html>`__

First of all you need to activate your virtual environment. This is accomplished using one of the two following commands:

.. code:: bash

    # Either
    poetry shell
    # or
    source recommender/venv/recommender/bin/activate

In the virtual environment, you can now run your calculations by executing the `main<./../src/asme/main.py>`__ method.

.. code:: bash

    python main.py [COMMAND] [ARGS]


Possible commands include:

*  train
*  resume
*  search
*  predict
*  evaluate


Further information about these commands are listed below.

Training Implemented Models
---------------------------

Run Training
""""""""""""""

    python main.py train <config_file> [OPTIONAL ARGS]

+---------------------------------+--------------------------------------------------------------------------------+---------------+
| Arg (Type)                      | Description                                                                    | Default Value |
+=================================+================================================================================+===============+
| config_file (Path)              | path to the config file                                                        |               |
+---------------------------------+--------------------------------------------------------------------------------+---------------+
| do_resume (bool)                | if you want to resume training from existing checkpoint                        | False         |
+---------------------------------+--------------------------------------------------------------------------------+---------------+
| print_train_val_examples (bool) | print examples of training and evaluation dataset before starting the training | True          |
+---------------------------------+--------------------------------------------------------------------------------+---------------+

Further information about the structure of the config file are listed `here <./configuration.html>`__

Resume Training
""""""""""""""""
As an alternative to resuming training by flagging do_resume, you can run resume directly:

.. code:: bash

    python main.py resume <log_dir> <checkpoint_file>

+---------------------------------+-------------------------------------------+---------------+
| Arg (Type)                      | Description                               | Default Value |
+=================================+===========================================+===============+
| log_dir (str)                   | path to logging directory                 |               |
+---------------------------------+-------------------------------------------+---------------+
| checkpoint_file (str)           | path to checkpoint file to resume from    | None          |
+---------------------------------+-------------------------------------------+---------------+

Executing Trained Models
------------------------

Hyperparameter Search
"""""""""""""""""""""
Searching for hyperparameter with Optuna.
Further information can be found `here <./hyperparameter_search.html>`__

.. code:: bash

    python main.py search <template_file> <study_name> <study_storage> <objective_metric> [ARGS]


+---------------------------------+--------------------------------------------------------------------------------+---------------+
| Arg (Type)                      | Description                                                                    | Default Value |
+=================================+================================================================================+===============+
| template_file (Path)            | path to the config file                                                        |               |
+---------------------------------+--------------------------------------------------------------------------------+---------------+
| study_name (str)                | study name of an existing Optuna study                                         |               |
+---------------------------------+--------------------------------------------------------------------------------+---------------+
| study_storage (str)             | connection string for the study storage                                        |               |
+---------------------------------+--------------------------------------------------------------------------------+---------------+
| objective_metric (str)          | name of the metric to watch during study (e.g. recall@5)                       |               |
+---------------------------------+--------------------------------------------------------------------------------+---------------+
| study_direction (str)           | minimize / maximize                                                            | maximize      |
+---------------------------------+--------------------------------------------------------------------------------+---------------+
| num_trails (int)                | number of trails to execute                                                    | 20            |
+---------------------------------+--------------------------------------------------------------------------------+---------------+

Prediction
""""""""""

.. code:: bash

    python main.py predict <output_file> [ARGS]

+-------------------------------+-----------------------------------------+---------------+
| Arg (Type)                    | Description                             | Default Value |
+===============================+=========================================+===============+
| output_file (Path)            | path where output is written            |               |
+-------------------------------+-----------------------------------------+---------------+
| num_predictions (int)         | number of predictions to export         | 20            |
+-------------------------------+-----------------------------------------+---------------+
| gpu (int)                     | number of gpus to use                   | 0             |
+-------------------------------+-----------------------------------------+---------------+
| selected_items_file (Path)    | only use these item ids for prediction  | None          |
+-------------------------------+-----------------------------------------+---------------+
| checkpoint_file (Path)        | path to the checkpoint file             | None          |
+-------------------------------+-----------------------------------------+---------------+
| config_file (Path)            | path to the config file                 | None          |
+-------------------------------+-----------------------------------------+---------------+
| study_name (str)              | name of an existing study               | None          |
+-------------------------------+-----------------------------------------+---------------+
| study_storage (str)           | connection string for the study storage | None          |
+-------------------------------+-----------------------------------------+---------------+
| overwrite (bool)              | overwrite output file (if it exists)    | False         |
+-------------------------------+-----------------------------------------+---------------+
| log_input (bool)              | enable input logging                    | True          |
+-------------------------------+-----------------------------------------+---------------+
| log_per_sample_metrics (bool) | enable logging of per-sample metrics    | True          |
+-------------------------------+-----------------------------------------+---------------+
| seed (int)                    | seed used e.g. for sampled evaluation   | None          |
+-------------------------------+-----------------------------------------+---------------+


Evaluation
""""""""""

.. code:: bash

    python main.py evaluate <config_file> <checkpoint_file> <study_name> <study_storage> [OPTIONAL ARGS]

+------------------------+-------------------------------------------+---------------+
| Arg (Type)             | Description                               | Default Value |
+========================+===========================================+===============+
| config_file (Path)     | path to the config file                   |               |
+------------------------+-------------------------------------------+---------------+
| checkpoint_file (Path) | path to the checkpoint file               |               |
+------------------------+-------------------------------------------+---------------+
| study_name (str)       | study name of an existing study           |               |
+------------------------+-------------------------------------------+---------------+
| study_storage (str)    | connection string for the study storage   |               |
+------------------------+-------------------------------------------+---------------+
| output_file (Path)     | path where output is written              | None          |
+------------------------+-------------------------------------------+---------------+
| gpu (int)              | number of gpus to use                     | 0             |
+------------------------+-------------------------------------------+---------------+
| overwrite (bool)       | overwrite output file (if it exists)      | False         |
+------------------------+-------------------------------------------+---------------+
| seed (int)             | seed used e.g. for the sampled evaluation | None          |
+------------------------+-------------------------------------------+---------------+

Using integrated Datasets
------------------------
ASME includes several datasets such as ml-1m and ml-20m that can be preprocessed in order to fit your needs.
The process of selecting a dataset and specifying its preprocessing is facilitated via corresponding entries in the configuration file.
The dataset can be choosen by setting the `name` parameter in the `datamodule` section of the configuration to the matching name.
Preprocessing can than be adapted to fit your needs by setting appropriate parameters in the `preprocessing` section of the `datamodule` configuration entry.


Generally, ASME will generate three types of splits for each dataset:

- Ratio split: The data is split by session into three sets for training, validation and testing.
- Leave-One-Out (LOO) split: The data is split intra-session, i.e. a session of n items is decomposed into a n - 2 item training session, a 1 item validation and a 1 item test session.
- Leave-Percentage-Out (LPO) split: Similar to the LOO split, the data is partioned intra-session. Specifically, the first t% of each session is used for training, the next v% for validation and the final (1 - t - v)% for testing.


You can customize the preprocessing process by setting the following parameters:


- Ratio split:
    - ratio_split_min_item_feedback:        The minimum number of interactions necessary for an item to be kept in the dataset.
    - ratio_split_min_sequence_length:      The minimum number of interactions in a session for it to be kept in the dataset.
    - ratio_split_train_percentage:         The fraction of session to (approximately) include into the training set.
    - ratio_split_validation_percentage:    The fraction of session to (approximately) include into the validation set.
    - ratio_split_test_percentage:          The fraction of session to (approximately) include into the test set.
    - ratio_split_window_markov_length:     The size of the sliding window that is used to extract samples from each session.
    - ratio_split_window_target_length:     The size of the sliding window that is used to extract targets from each session.
    - ratio_split_session_end_offset:       The distance between the session end and the last position for the sliding window.

- Leave-One-Out split:
    - loo_split_min_item_feedback:      The minimum number of interactions necessary for an item to be kept in the dataset.
    - loo_split_min_sequence_length:    The minimum number of interactions in a session for it to be kept in the dataset.

- Leave-Percentage-Out split:
    - lpo_split_min_item_feedback:      The minimum number of interactions necessary for an item to be kept in the dataset.
    - lpo_split_min_sequence_length:    The minimum number of interactions in a session for it to be kept in the dataset.
    - lpo_split_train_percentage:       The fraction of session to (approximately) include into the training set.
    - lpo_split_validation_percentage:  The fraction of session to (approximately) include into the validation set.
    - lpo_split_test_percentage:        The fraction of session to (approximately) include into the test set.
    - lpo_split_min_train_length:       The minimum size of each session in the training set.
    - lpo_split_min_validation_length:  The minimum size of each session in the validation set.
    - lpo_split_min_test_length:        The minimum size of each session in the test set.
