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

RUN mkdir -p /root/local/src \
    && cd /root/local/src \
    && git clone https://aomedia.googlesource.com/aom \
    && cd aom \
    && cd build \
    && cmake .. \
    && make \
    && make install

RUN mkdir -p /root/local/src \
    && cd /root/local/src \
    && git clone https://chromium.googlesource.com/libyuv/libyuv \
    && cd libyuv \
    && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE="Release" .. \
    && cmake --build . --config Release \
    && cmake --build . --target install --config Release

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpng-dev \
        libjpeg-dev

RUN mkdir -p /root/local/src \
    && cd /root/local/src \
    && git clone https://github.com/AOMediaCodec/libavif.git -b v0.8.4 \
    && cd libavif \
    && git submodule update --init --recursive

# cd /root/local/src/libavif/ && mkdir build && cd build && cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DAVIF_CODEC_AOM=ON -DAVIF_BUILD_TESTS=1 -DAVIF_BUILD_APPS=1
