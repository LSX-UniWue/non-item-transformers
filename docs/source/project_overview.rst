.. _project_overview:

The-Jane-Doe-Framework
=======================
(Name subject to change)

The Jane-Doe-Framework is a project which aims at alleviating the lack
of comparison in research literature concerning recommendation systems.

The idea behind the project is to provide popular metrics, data sets and
recommender models in one framework.

Thereby, interfaces are unified for all models and it is possible to use
the same metrics and data sets to train and test models. This makes
comparisons between old and new models a lot easier.

In order to realize this project multiple metrics are implemented as
well as the pre-processing of multiple data sets. Additionally popular
baseline models like SasRec and Bert4Rec are implemented and trained for
comparison.

Techstack
---------

This project uses:

- `Pytorch Lightning <https://www.pytorchlightning.ai/>`__ for deep-learning model implementation and training
- `Typer <https://typer.tiangolo.com/>`__ for executing python code as a CLI
- `Poetry <https://python-poetry.org/docs/#installation>`__ for dependency management

Getting Started
---------------

This section describes how to setup your python environment and execute your first model.

Setup Environment
~~~~~~~~~~~~~~~~

Prerequisites:

- Generated and added an SSH-Key to your GitLab account. (Find out how: https://docs.gitlab.com/ee/ssh/#adding-an-ssh-key-to-your-gitlab-account)

First, clone the git repository.

.. code:: bash

    git clone git@gitlab2.informatik.uni-wuerzburg.de:dmir/dallmann/recommender.git

Second, install
`Poetry <https://python-poetry.org/docs/#installation>`__.

.. code:: bash

    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

Third, build the `virtual
environment <https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/>`__
using `Poetry <https://python-poetry.org/docs/#installation>`__.

.. code:: bash

    cd recommender 
    python3 -m venv venv/recommender
    source venv/recommender/bin/activate
    pip install poetry
    poetry install

If you are interested in how to use the framework continue to the `User Guide <./user_guide.html>`__.

If you are interested in the development of the framework continue to
the `Developer Guide <./developer_guide.html>`__.
