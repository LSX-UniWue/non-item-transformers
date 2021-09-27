FROM docker.io/pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# install some basic utilities
RUN apt-get update && apt-get -y install zsh byobu tmux curl htop vim git wget mc build-essential xz-utils rsync && rm -rf /var/lib/apt/lists/*

# install poetry into /opt/poetry
ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
ENV PATH="/opt/poetry/bin:${PATH}"
RUN chmod o+rx /opt/poetry/bin/poetry

ENV PYTHONUNBUFFERED=1
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV REPO_USER=dallmann
ENV REPO_BRANCH=master
ENV PROJECT_DIR=/project

COPY .git-askpass /.git-askpass
RUN chmod +x /.git-askpass

ENV GIT_TOKEN="123456"
ENV GIT_ASKPASS=/.git-askpass

COPY entrypoint.sh /entrypoint.sh

RUN chmod ugo+rx /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh" ]
