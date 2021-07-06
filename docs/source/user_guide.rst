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

Yoochoose Data Set
~~~~~~~~~~~~~~~~~~

Pre-Requisites:
- Downloaded the `yoochoose data set <https://www.kaggle.com/chadgostopp/recsys-challenge-2015/download>`__

Training Implemented Models
---------------------------

Executing Trained Models
------------------------

