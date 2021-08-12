.. _project_overview:

Project Overview
=======================

The Jane-Doe-Framework is a project which aims at alleviating the lack
of comparison in research literature concerning recommendation systems.

The idea behind the project is to provide popular metrics, data sets and
recommender models in one framework.

Thereby, all models have unified interfaces and it is possible to use
the same metrics and data sets to train and test them. This makes
comparisons between old and new models a lot easier.

In order to realize this project, multiple metrics are implemented as
well as the pre-processing of several data sets. Additionally popular
baseline models like SasRec and Bert4Rec are included and trained for
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

Environment Setup
~~~~~~~~~~~~~~~~~~
- First, it is required to generate a SSH-Key and add it to your GitLab account. (Learn more: https://docs.gitlab.com/ee/ssh/#adding-an-ssh-key-to-your-gitlab-account)

- Second, clone the git repository.

.. code:: bash

    git clone git@gitlab2.informatik.uni-wuerzburg.de:dmir/dallmann/recommender.git

- Third, install Poetry. Find out about system-specific installation in its `documenation <https://python-poetry.org/docs/#installation>`__.

- Finally, build the `virtual environment <https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/>`__ using `Poetry <https://python-poetry.org/docs/#installation>`__.

.. code:: bash

    cd recommender 
    python3 -m venv venv/recommender
    source venv/recommender/bin/activate
    pip install poetry
    poetry install

Find out more about how to use the framework in the `User Guide <./user_guide.html>`__.

If you are interested in the development of the framework go to the `Developer Guide <./developer_guide.html>`__.
