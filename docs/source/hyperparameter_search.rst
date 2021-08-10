.. _hyperparameter_search:

Hyperparameter Search
=====================

We use Optuna to optimize the hyperparameters of the models in the
framework. Optional you can use MLFlow to track your current parameter
search.

1. Step: Setup Infrastructure
-----------------------------

Optuna Storage
~~~~~~~~~~~~~~

Setup a storage for optuna.

There are kubernetes examples for redis in this repo.

MLFlow
~~~~~~

Setup mflow, configure it in the run config

2. Step: Create Optuna Study
----------------------------

(This is optional, by default the first hyperopt search will create a
study provided with the study name) Create a study using your favorite
storage backend.

::

    optuna create-study --study-name $STUDY_NAME --storage $STORAGE --direction $DIRECTION

e.g. storage can be ``redis://redis``, direction can be
``{minimize,maximize}`` and should be ``maximize`` for recall@k

Copy the study name from the cli output of optuna.

3. Step: Create Run Config
--------------------------

Create a run config.

Instead of config a fixed value for a hyperparameter:

.. code:: json

    model: {
        ...
        num_transformer_heads: 4,
        ...
    }

Add a hyper\_opt config object to the hyperparameter:

.. code:: json

    model: {
        ...
        transformer_hidden_size: {
                hyper_opt: {
                    suggest: "int",
                    params: {
                        low: 2,
                        high: 8,
                        step: 2
                    }
                }
            }
        },
        ...
    }

Possible values for the suggest function are:

-  `categorical <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_categorical>`__
-  `discrete\_uniform <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_discrete_uniform>`__
-  `float <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_float>`__
-  `int <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_int>`__
-  `loguniform <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_loguniform>`__
-  `uniform <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_uniform>`__

Please refer to the Optuna documentation for the available parameters
for each suggest function.

If a hyperparameter depends on another hyperparameter you can specify
this also in the config:

.. code:: json

    model {
        ...
        transformer_hidden_size: {
            hyper_opt: {
              suggest: "int",
              params: {
                low: 2
                high: 8
                step: 2
              },
              dependency: {
                type: "multiply",
                on: "model.num_transformer_heads"
              }
            }
        },
        ...
    }

Currently, we support the following dependencies:

-  multiply: the suggested value is multiplied with the dependent value
-  optimize\_iff: the hyperparameter will only be generated if a
   condition is satisfied based on another hyperparameter

.. code:: json

    model {
        ...
        transformer_hidden_size: {
            hyper_opt: {
              suggest: "int",
              params: {
                low: 2
                high: 8
                step: 2
              },
              dependency: {
                on: "module.model.num_transformer_heads",
                type: "optimize_iff",
                conditions: [{
                  type: 'equal',
                  compare_value: 5
                }]
              }
            }
        },
        ...
    }

4. Step: Run Study
------------------

