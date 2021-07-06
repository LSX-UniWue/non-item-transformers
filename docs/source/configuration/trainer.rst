.. _config trainer:
Trainer
=======

Maximum number of epochs:
------------------------
.. code::json

    ...
    trainer: {
        ...
        max_epochs: 5
    }

Early Stopping:
---------------

If you want to early stop training based on a metric or the loss, add
the following to the trainer config:

.. code:: json

    ...
    trainer: {
        ...
        early_stopping: {
          monitor: METRIC_OR_LOSS,
          min_delta: MIN_DELTA,
          patience: PATIENCE,
          mode: MODE
        },
        ...
    },
    ...


See `Pytorch Lightning
Docu <https://pytorch-lightning.readthedocs.io/en/stable/common/early_stopping.html?highlight=early%20stopping#early-stopping-based-on-metric-using-the-earlystopping-callback>`__
for more details.

Checkpoints
-----------

.. code:: json:

    ...
    trainer: {
        ...
        checkpoint: {
            monitor: "recall@5",
            save_top_k: 3,
            mode: 'max'
        },
        ...
    },

Loggers
-------

Tensorboard
~~~~~~~~~~~
.. code:: json

    ...
    trainer: {
        ...
        loggers: {
            ...
            tensorboard: {},
            ...
        },
        ...
    },
    ...


CSV
~~~

The CSV logger logs the hyperparameters into a hparams.yaml file and the
metrics into the metrics.csv file under the save\_dir.

.. code:: json

    ...
    trainer: {
        ...
        loggers: {
            ...
            csv: {},
            ...
        },
        ...
    },
    ...

See `PyTorch Lightning Docu for
CSVLogger <https://pytorch-lightning.readthedocs.io/en/0.9.0/api/pytorch_lightning.loggers.csv_logs.html>`__
for all parameters that can be configured.

mlflow
~~~~~~

Under trainer add a logger section:

.. code:: json

    ...
    trainer: {
        ...
        loggers: {
            ...
            mlflow {
                experiment_name: "test",
                tracking_uri: "http://localhost:5000"
            },
            ...
        }
        ...
    },
    ...

wandb
~~~~~

Under trainer add a logger section:

.. code:: json

    ...
    trainer {
        ...
        loggers: {
            ...
            "wandb": {
                log_model: false,
                project: "test"
            },
            ...
        },
        ...
    },
    ...

