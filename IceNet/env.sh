#! /usr/bin/env bash

ENV_NAME=$1
ENV_SPEC=$2

function create_env {
	echo "creating $ENV_NAME from $ENV_SPEC"
	conda env create -n $ENV_NAME
    conda env update -n $ENV_NAME --file $ENV_SPEC
	eval "$(conda shell.bash hook)" && conda activate $ENV_NAME
	conda install ipykernel -y
	python -m ipykernel install --user --name $ENV_NAME --display-name $ENV_NAME
    jupyter nbextension enable --py widgetsnbextension
}

# Code below mostly from stackoverflow
# https://stackoverflow.com/questions/60115420/check-for-existing-conda-environment-in-makefile

RED='\033[1;31m'
GREEN='\033[1;32m'
CYAN='\033[1;36m'
NC='\033[0m' # No Color

if ! (return 0 2>/dev/null) ; then
    # If return is used in the top-level scope of a non-sourced script,
    # an error message is emitted, and the exit code is set to 1
    echo
    echo -e $RED"This script should be sourced like"$NC
    echo "    . ./activate.sh"
    echo
    exit 1  # we detected we are NOT source'd so we can use exit
fi

if type conda 2>/dev/null; then
    if conda info --envs | grep ${ENV_NAME}; then
      echo -e $CYAN"activating environment ${ENV_NAME}"$NC
    else
      echo
      echo -e $RED"(!) Will install the conda environment ${ENV_NAME}"$NC
      echo
      create_env
      return 1  # we are source'd so we cannot use exit
    fi
else
    echo
    echo -e $RED"(!) Please install anaconda"$NC
    echo
    return 1  # we are source'd so we cannot use exit
fi

eval "$(conda shell.bash hook)" && conda activate $ENV_NAME
echo -e $RED"Change kernel to $ENV_NAME, refresh browser if not available."