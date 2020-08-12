FROM jupyter/scipy-notebook

RUN mkdir -p /home/jovyan/work \
    && mkdir -p /home/jovyan/abuse

RUN conda install -c conda-forge --quiet --yes \
    opencv \
    colour-science

RUN conda install --quiet --yes \
    flake8

RUN conda install -c plotly --quiet --yes \
    plotly=4.9.0 \
    && \
    conda clean --all -f -y

RUN jupyter labextension install jupyterlab-plotly@4.9.0
