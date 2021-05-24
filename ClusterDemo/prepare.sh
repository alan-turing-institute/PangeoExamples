#!/bin/bash
# Modified from https://github.com/dask/dask-docker
# Copyright (c) 2019, Dask Developers. All rights reserved.
# See license https://github.com/dask/dask-docker/blob/main/LICENSE

# We start by adding extra apt packages, since pip modules may required library
if [ "$EXTRA_APT_PACKAGES" ]; then
    echo "EXTRA_APT_PACKAGES environment variable found.  Installing."
    apt update -y
    apt install -y $EXTRA_APT_PACKAGES
fi

# if [ -e "/opt/app/environment.yml" ]; then
#     echo "environment.yml found. Installing packages"
#     /opt/conda/bin/mamba env update -n base -f /opt/app/environment.yml
# else
#     echo "no environment.yml"
# fi

if [ "$EXTRA_CONDA_PACKAGES" ]; then
    echo "EXTRA_CONDA_PACKAGES environment variable found.  Installing."
    /opt/conda/bin/mamba install -y $EXTRA_CONDA_PACKAGES
fi

if [ "$EXTRA_PIP_PACKAGES" ]; then
    echo "EXTRA_PIP_PACKAGES environment variable found.  Installing".
    /opt/conda/bin/pip install $EXTRA_PIP_PACKAGES
fi

# Run extra commands
exec "$@"

