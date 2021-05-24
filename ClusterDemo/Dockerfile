FROM continuumio/miniconda3:4.8.2

RUN mkdir /opt/app

COPY prepare.sh /usr/bin/prepare.sh
COPY environment.yml /opt/app/environment.yml

RUN conda install mamba -n base -c conda-forge

RUN /opt/conda/bin/mamba env update -n base -f /opt/app/environment.yml
RUN mamba install --yes \
    -c conda-forge \
    python==3.8 \
    python-blosc \
    cytoolz \
    dask==2021.4.1 \
    lz4 \
    nomkl \
    numpy==1.18.1 \
    pandas==1.0.1 \
    tini==0.18.0 \
    && mamba clean -tipsy \
    && find /opt/conda/ -type f,l -name '*.a' -delete \
    && find /opt/conda/ -type f,l -name '*.pyc' -delete \
    && find /opt/conda/ -type f,l -name '*.js.map' -delete \
    && find /opt/conda/lib/python*/site-packages/bokeh/server/static -type f,l -name '*.js' -not -name '*.min.js' -delete \
    && rm -rf /opt/conda/pkgs

ENTRYPOINT ["tini", "-g", "--", "/usr/bin/prepare.sh"]

