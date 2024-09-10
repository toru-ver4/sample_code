ARG DEBIAN_VERSION="bookworm"
ARG PYTHON_VERSION="3.12"

FROM python:${PYTHON_VERSION}-${DEBIAN_VERSION} as builder

# Avoid clearing `DEBIAN_VERSION`
ARG DEBIAN_VERSION
ARG PYTHON_VERSION

WORKDIR /root/local

RUN echo "deb http://ftp.jp.debian.org/debian/ ${DEBIAN_VERSION} main" > /etc/apt/sources.list \
    && echo "deb http://security.debian.org/ ${DEBIAN_VERSION}-security main" >> /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libx265-dev \
        libde265-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install libheif
RUN git clone https://github.com/strukturag/libheif.git -b v1.18.2 \
    && cd libheif \
    && mkdir build \
    && cd build \
    && cmake --preset=release .. \
    && make -j16 \
    && make install \
    && cd /root/local \
    && rm -rf /root/local/libheif

RUN ldconfig

CMD ["bash"]
