#!/usr/bin/env bash
REPO_URL="https://${REPO_USER}@gitlab2.informatik.uni-wuerzburg.de/dmir/dallmann/recommender.git"

# create project dir
mkdir -p "${PROJECT_DIR}"
cd "${PROJECT_DIR}"
# checkout the repository

# TODO: recommender
if [ ! -d "${PROJECT_DIR}/recommender" ]; then
    git clone -q "${REPO_URL}"
fi

cd recommender || exit 1
git checkout -q "${REPO_BRANCH}"

# update the repo if already checked out
git pull -q

# install requirements using poetry
poetry install

# execute the provided prepare script
if [ "$PREPARE_SCRIPT" ]; then
  echo "x${PREPARE_SCRIPT}x"
  chmod +x "${PREPARE_SCRIPT}"
  $PREPARE_SCRIPT
fi

# now run the configuration using poetry

export PYTHONPATH=${PROJECT_DIR}/recommender
/bin/bash -c "cd ${PROJECT_DIR}/recommender && poetry run python -m plp_paper.plp_main $*"