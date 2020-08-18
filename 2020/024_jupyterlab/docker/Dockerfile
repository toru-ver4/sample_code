FROM jupyter/scipy-notebook

RUN mkdir -p /home/jovyan/work \
    && mkdir -p /home/jovyan/abuse

RUN conda install -c conda-forge --quiet --yes \
    opencv \
    colour-science \
    vispy

RUN conda install --quiet --yes \
    flake8

ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"

USER root

RUN apt-get update && apt-get -y install \
    qt5-default \
    qtbase5-dev \
    qttools5-dev-tools

RUN conda install -c conda-forge --quiet --yes \
    pyqt \
    nodejs \
    ipympl=0.5.3

RUN jupyter lab build -y

RUN fix-permissions "/home/jovyan/abuse"

USER $NB_UID
