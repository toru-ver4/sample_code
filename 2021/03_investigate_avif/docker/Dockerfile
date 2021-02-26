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

# avif with aom
RUN mkdir -p /root/local/src \
    && cd /root/local/src \
    && git clone https://github.com/AOMediaCodec/libavif.git -b v0.8.4 libavif_aom \
    && cd /root/local/src/libavif_aom/ext \
    && $SHELL ./aom.cmd \
    && cd /root/local/src/libavif_aom/ \
    && mkdir build && cd build \
    && cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DAVIF_CODEC_AOM=ON -DAVIF_LOCAL_AOM=ON -DAVIF_BUILD_TESTS=1 -DAVIF_BUILD_APPS=1 \
    && ninja \
    && cp avifdec /usr/bin/avifdec_aom \
    && cp avifenc /usr/bin/avifenc_aom

# avif with DAV1D
RUN mkdir -p /root/local/src \
    && cd /root/local/src \
    && git clone https://github.com/AOMediaCodec/libavif.git -b v0.8.4 libavif_dav1d \
    && cd /root/local/src/libavif_dav1d/ext \
    && $SHELL ./dav1d.cmd \
    && cd /root/local/src/libavif_dav1d/ \
    && mkdir build && cd build \
    && cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DAVIF_CODEC_DAV1D=ON -DAVIF_LOCAL_DAV1D=ON -DAVIF_BUILD_TESTS=1 -DAVIF_BUILD_APPS=1 \
    && ninja \
    && cp avifdec /usr/bin/avifdec_dav1d \
    && cp avifenc /usr/bin/avifenc_dav1d

RUN apt-get update && apt-get install -y --no-install-recommends \
        cbindgen \
        cargo

# avif with RAV1E
RUN mkdir -p /root/local/src \
    && cd /root/local/src \
    && git clone https://github.com/AOMediaCodec/libavif.git -b v0.8.4 libavif_rav1e \
    && cd /root/local/src/libavif_rav1e/ext \
    && $SHELL ./rav1e.cmd \
    && cd /root/local/src/libavif_rav1e/ \
    && mkdir build && cd build \
    && cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DAVIF_CODEC_RAV1E=ON -DAVIF_LOCAL_RAV1E=ON -DAVIF_BUILD_TESTS=1 -DAVIF_BUILD_APPS=1 \
    && ninja \
    && cp avifdec /usr/bin/avifdec_rav1e \
    && cp avifenc /usr/bin/avifenc_rav1e

# avif with libgav1
RUN mkdir -p /root/local/src \
    && cd /root/local/src \
    && git clone https://github.com/AOMediaCodec/libavif.git -b v0.8.4 libavif_libgav1 \
    && cd /root/local/src/libavif_libgav1/ext \
    && $SHELL ./libgav1.cmd \
    && cd /root/local/src/libavif_libgav1/ \
    && mkdir build && cd build \
    && cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DAVIF_CODEC_LIBGAV1=ON -DAVIF_LOCAL_LIBGAV1=ON -DAVIF_BUILD_TESTS=1 -DAVIF_BUILD_APPS=1 \
    && ninja \
    && cp avifdec /usr/bin/avifdec_libgav1 \
    && cp avifenc /usr/bin/avifenc_libgav1

# avif with svt
RUN mkdir -p /root/local/src \
    && cd /root/local/src \
    && git clone https://github.com/AOMediaCodec/libavif.git -b v0.8.4 libavif_svt \
    && cd /root/local/src/libavif_svt/ext \
    && chmod 755 ./svt.sh \
    && $SHELL ./svt.sh \
    && cd /root/local/src/libavif_svt/ \
    && mkdir build && cd build \
    && cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DAVIF_CODEC_SVT=ON -DAVIF_LOCAL_SVT=ON -DAVIF_BUILD_TESTS=1 -DAVIF_BUILD_APPS=1 \
    && ninja \
    && cp avifdec /usr/bin/avifdec_svt \
    && cp avifenc /usr/bin/avifenc_svt

# RUN rm -rf /root/local/src \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
