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

The components of this framework can be executed using the `Runner <../asme/runner>`__.

In the virtual environment, you can now run your calculations by executing the `main<./../src/asme/main.py>`__ method.

.. code:: bash
    python main.py [COMMAND] [ARGS]

Possible commands include: 
- train
- resume
- search
- predict
- evaluate


Further information about these commands are listed below.

Training Implemented Models
---------------------------
.. code:: bash
    python main.py train <path_to_config_file> [ARGS]


Arguments:
- config_file (Path, must exist): path to the config file
- do_resume (bool False): if you want to resume training
- print_train_val_examples (bool True): print examples of training and evaluation dataset before starting the training
Further information about the structure of the config file are listed at `Configurations<./configuration.html`__.

Alternatively to resuming training by flagging do_resume, there's a resume option:
.. code:: bash 
    python main.py resume <log_dir> [checkpoint_file]

Arguments:
- log_dir (str, must exist): path to logging directory
- checkpoint_file (str): path to checkpoint file to resume from

Executing Trained Models
------------------------
Searching for hyperparameter with Optuna

.. code:: bash 
    python main.py search <template_file> <study_name> <study_storage> <objective_metric> [ARGS]

Arguments:
- template_file: path to config file
- study_name: study name of an existing optuna study
- study_storage: connection string for the study storage
- objective_metric: the name of the metric to watch during the study (e.g. recall@5)
- study_direction: minimize or maximize, default = maximize
- num_trails: number of trails to execute (defaut = 20)
           


Predicting

.. code:: bash
    python main.py predict <output_file> [ARGS]

Arguments:
- output_file: path where output is written
- num_predictions: number of predictions to export, default=20
- gpu: number of gpus to use, default=0
- selected_items_file: only use these item ids for prediction
- checkpoint_file: path to the checkpoint file
- config_file: path to the config file
- study_name: name of an existing study
- study_storage: connection string for the study storage
- overwrite: overwrite output file (if it exists), default = False
- log_input: enable input logging, defaut = True
- log_per_sample_metrics: enable logging of per-sample metrics, default=True
- seed: seed used e.g. for the sampled evaluation


Evaluation

.. code:: bash  
    python main.py evaluate 

- config_file: path to the config file
- checkpoint_file: path to the checkpoint file
- study_name: study name of an existing study
- study_storage: connection string for the study storage
- output_file: path where output is written
- gpu: number of gpus to use
- overwrite: overwrite output file (if it exists), default = False
- seed: seed used e.g. for the sampled evaluation
             


Below are (old) instructions for pre-processing a dataset:

Pre-Processing Data Sets
------------------------

For all data sets a CLI is provided via `Typer <https://typer.tiangolo.com/>`__.

MovieLens Data Set
~~~~~~~~~~~~~~~~~~
To download and pre-process the MovieLens data set use the following commands:

.. code:: bash

    python -m dataset.movielens ml-1m
    python -m runner.dataset.create_reader_index ./dataset/ml-1m_5/ml-1m.csv ./dataset/ml-1m_5/index.csv --session_key userId
    python -m runner.dataset.create_csv_dataset_splits ./dataset/ml-1m_5/ml-1m.csv ./dataset/ml-1m_5/index.csv ./dataset/ml-1m_5/splits/ "train;0.9" "valid;0.05" "test;0.05"
    python -m runner.dataset.create_next_item_index ./dataset/ml-1m_5/splits/test.csv ./dataset/ml-1m_5/index.csv ./dataset/ml-1m_5/splits/test.nip.csv movieId

This downloads the MovieLens data set and prepares the data split for next item recommendation.

YooChoose Data Set
~~~~~~~~~~~~~~~~~~

Pre-Requisites:
- Downloaded the `yoochoose data set <https://www.kaggle.com/chadgostopp/recsys-challenge-2015/download>`__
