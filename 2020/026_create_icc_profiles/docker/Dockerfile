FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    autoconf \
    libtool \
    libxml2-dev \
    libtiff5-dev

RUN mkdir -p /root/local/src
COPY SampleICC-1.6.8.tar.gz /root/local/src/
COPY IccXML-0.9.8.tar.gz /root/local/src/

RUN mkdir -p /work/src

RUN cd /root/local/src \
    && tar -xzf SampleICC-1.6.8.tar.gz \
    && cd SampleICC-1.6.8 \
    && ./configure \
    && make -j8 \
    && make install

RUN cd /root/local/src \
    && tar -xzf IccXML-0.9.8.tar.gz \
    && cd IccXML-0.9.8 \
    && ./configure \
    && make -j8 \
    && make install

RUN echo "/usr/local/lib" >> etc/ld.so.conf \
    && ldconfig

RUN rm -rf /root/local/src \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
