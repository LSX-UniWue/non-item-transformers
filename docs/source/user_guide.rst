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
- train
- resume
- search
- predict
- evaluate


Further information about these commands are listed below.

Training Implemented Models
---------------------------

Run Training
""""""""""""""
.. code:: bash
    python main.py train <config_file> [OPTIONAL ARGS]

with
- config_file (Path): path to the config file
Optional arguments:
- do_resume (bool, default=False): if you want to resume training from a checkpoint
- print_train_val_examples (bool, default=True): print examples of training and evaluation dataset before starting the training

Further information about the structure of the config file are listed at `Configurations<./configuration.html`__.

Resume Training
""""""""""""""""
As an alternative to resuming training by flagging do_resume, you can run resume directly:
.. code:: bash 
    python main.py resume <log_dir> <checkpoint_file>

with
- log_dir (str): path to logging directory
Optional arguments:
- checkpoint_file (str): path to checkpoint file to resume from

Executing Trained Models
------------------------

Hyperparameter Search
"""""""""""""""""""""
Searching for hyperparameter with Optuna

.. code:: bash 
    python main.py search <template_file> <study_name> <study_storage> <objective_metric> [ARGS]

with
- template_file: path to config file
- study_name: study name of an existing Optuna study
- study_storage: connection string for the study storage
- objective_metric: the name of the metric to watch during the study (e.g. recall@5)
- study_direction: minimize or maximize, default = maximize
- num_trails: number of trails to execute (defaut = 20)

Prediction
""""""""""

.. code:: bash
    python main.py predict <output_file> [ARGS]

with
- output_file: path where output is written
Optional arguments:
- num_predictions: number of predictions to export, default = 20
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
""""""""""

.. code:: bash  
    python main.py evaluate <config_file> <checkpoint_file> <study_name> <study_storage> [OPTIONAL ARGS]

with
- config_file: path to the config file
- checkpoint_file: path to the checkpoint file
- study_name: study name of an existing study
- study_storage: connection string for the study storage
Optional arguments:
- output_file: path where output is written
- gpu: number of gpus to use
- overwrite: overwrite output file (if it exists), default = False
- seed: seed used e.g. for the sampled evaluation
