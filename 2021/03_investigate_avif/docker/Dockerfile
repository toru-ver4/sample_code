FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# build tools for vmaf
RUN apt-get update && \
	apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ninja-build \
        doxygen \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        python3-tk \
        yasm \
        cmake \
        nasm

RUN mkdir -p /root/local/src
RUN mkdir -p /root/local/src \
    && cd / \
    && git clone https://github.com/Netflix/vmaf.git \
    && cd vmaf

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade cython numpy
RUN pip3 install --no-cache-dir meson

ENV PATH=/vmaf:/vmaf/libvmaf/build/tools:$PATH
RUN cd /vmaf && make

RUN mkdir -p /work/src
RUN mkdir -p /root/local/src \
    && cd /root/local/src \
    && git clone --recurse-submodules --recursive https://github.com/link-u/cavif.git

RUN cd /root/local/src/cavif \
    && mkdir build && cd build \
    && cmake .. \
    && make cavif \
    && make install

RUN rm -rf /root/local/src \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
