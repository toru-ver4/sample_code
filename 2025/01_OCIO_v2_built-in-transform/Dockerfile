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
        clang \
        cmake \
        doxygen \
        ffmpeg \
        git \
        imagemagick \
        libboost-all-dev \
        libde265-dev \
        libexpat1-dev \
        libfreetype-dev \
        libgif-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libheif-dev \
        libimath-dev \
        libjpeg-dev \
        liblcms2-dev \
        libopencv-dev \
        libopenexr-dev \
        libopenjp2-7-dev \
        libopenvdb-dev \
        libpng-dev \
        libpystring-dev \
        libtbb-dev \
        libtiff-dev \
        libraw-dev \
        libwebp-dev \
        libx265-dev \
        libyaml-cpp-dev \
        llvm \
        ninja-build \
        pybind11-dev \
        yasm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install OpencolorIO
RUN mkdir /tmp/ociobuild \
        && cd /tmp/ociobuild \
        && git clone https://github.com/AcademySoftwareFoundation/OpenColorIO.git /root/local/OpenColorIO -b v2.4.1 \
        && cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DOCIO_INSTALL_EXT_PACKAGES=MISSING -DOCIO_BUILD_PYTHON=1 -DOCIO_VERBOSE=1 /root/local/OpenColorIO \
        && make -j16 \
        && make install \
        && cd /root/local \
        && rm -rf /root/local/OpenColorIO /tmp/ociobuild

# Install OpenImageIO
RUN git clone https://github.com/OpenImageIO/oiio.git -b v3.0.2.0 \
        && cd oiio \
        && mkdir build \
        && cd build \
        && cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DPYTHON_VERSION=${PYTHON_VERSION} -DUSE_PYTHON=1 -DVERBOSE=1 -DSTOP_ON_WARNING=0 .. \
        && make install -j16 \
        && cd /root/local \
        && rm -rf /root/local/oiio
    
# Git init settings
RUN git config --global user.name toru.yoshihara \
    && git config --global user.email dummy_address@gmail.com \
    && git config --global core.autocrlf true

RUN ldconfig

CMD ["bash"]
