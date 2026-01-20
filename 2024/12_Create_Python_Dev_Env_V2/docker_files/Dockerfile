ARG PYTHON_VERSION="3.13"
ARG IMAGE_TYPE="slim"
ARG DEBIAN_VERSION="trixie"

FROM python:${PYTHON_VERSION}-${IMAGE_TYPE}-${DEBIAN_VERSION} AS builder

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
        libpystring-dev \
        libsvtav1enc-dev \
        libtbb-dev \
        libtiff-dev \
        libraw-dev \
        libwebp-dev \
        libx265-dev \
        libxcb-cursor0 \
        libx11-xcb1 \
        libxcb1 \
        libxcb-shm0 \
        libxcb-randr0 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-icccm4 \
        libxcb-xfixes0 \
        libxcb-render-util0 \
        libxkbcommon0 \
        libxkbcommon-x11-0 \
        libxml2-dev \
        libxml2-utils \
        libyaml-cpp-dev \
        llvm \
        ninja-build \
        pybind11-dev \
        python3-tk \
        tk-dev \
        sudo \
        unzip \
        wget \
        yasm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install lubultrahdr
########################################
# Note: Please modify `sed` command 
########################################
RUN git clone https://github.com/google/libultrahdr.git -b v1.4.0 \
    && cd libultrahdr \
    && sed -i '485,490s/^/\/\/ /' ./lib/src/ultrahdr_api.cpp \
    && mkdir build_directory \
    && cd build_directory \
    && cmake -G "Unix Makefiles" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DUHDR_BUILD_TESTS=1 ../ \
    && make -j16 \
    && cp ultrahdr_app /usr/local/bin/ \
    && cd / \
    && rm -rf /root/local/libultrahdr

COPY ./requirements.txt /root/local/requirements.txt

RUN pip install --upgrade pip \
        && pip install --no-cache-dir -r /root/local/requirements.txt

# Install libheif
RUN git clone https://github.com/strukturag/libheif.git -b v1.21.1 \
    && cd libheif \
    && mkdir build \
    && cd build \
    && cmake --preset=release .. \
    && make -j16 \
    && make install \
    && cd /root/local \
    && rm -rf /root/local/libheif

# Install OpencolorIO
RUN mkdir /tmp/ociobuild \
        && cd /tmp/ociobuild \
        && git clone https://github.com/AcademySoftwareFoundation/OpenColorIO.git /root/local/OpenColorIO -b v2.5.0 \
        && cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DOCIO_INSTALL_EXT_PACKAGES=MISSING -DOCIO_BUILD_PYTHON=0 -DOCIO_VERBOSE=1 /root/local/OpenColorIO \
        && make -j16 \
        && make install \
        && cd /root/local \
        && rm -rf /root/local/OpenColorIO /tmp/ociobuild

# Install OpenImageIO
RUN git clone https://github.com/OpenImageIO/oiio.git -b v3.1.8.0 \
        && cd oiio \
        && mkdir build \
        && cd build \
        && cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DPYTHON_VERSION=${PYTHON_VERSION} -DUSE_PYTHON=1 -DVERBOSE=1 -DSTOP_ON_WARNING=0 .. \
        && make install -j16 \
        && cd /root/local \
        && rm -rf /root/local/oiio
    
# Install CTL
RUN git clone https://github.com/ampas/CTL.git \
        && cd CTL \
        && mkdir build \
        && cd build \
        && cmake .. \
        && make -j16 \
        && make install \
        && cd /root/local \
        && rm -rf /root/local/CTL
    
# Install Sample ICC
RUN wget https://sourceforge.net/projects/sampleicc/files/sampleicc%20tar/SampleIcc-1.6.8/SampleICC-1.6.8.tar.gz \
        && tar xvzf SampleICC-1.6.8.tar.gz \
        && cd SampleICC-1.6.8 \
        && ./configure \
        && make -j16 \
        && make install \
        && cd /root/local \
        && rm -rf /root/local/SampleICC-1.6.8.tar.gz \
        && rm -rf /root/local/SampleICC-1.6.8
    
# Install ICC XML
RUN wget https://sourceforge.net/projects/iccxml/files/IccXML-Src/IccXML-0.9.8/IccXML-0.9.8.tar.gz \
        && tar -xzf IccXML-0.9.8.tar.gz \
        && cd IccXML-0.9.8 \
        && ./configure \
        && make -j16 \
        && make install \
        && cd /root/local \
        && rm -rf /root/local/IccXML-0.9.8.tar.gz \
        && rm -rf /root/local/IccXML-0.9.8

# Install libjxl (JPEG XL)
RUN echo "export CC=clang" >> /etc/profile \
    && echo "export CXX=clang++" >> /etc/profile \
    && git clone https://github.com/libjxl/libjxl.git --recursive --shallow-submodules \
    && cd libjxl \
    && git submodule update --init --recursive --depth 1 --recommend-shallow \
    && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF .. \
    && cmake --build . -- -j$(nproc) \
    && cmake --install . \
    && cd /root/local \
    && rm -rf /root/local/libjxl

# Install libavif
RUN git clone https://github.com/AOMediaCodec/libavif.git -b v1.3.0 libavif_aom \
    && cd libavif_aom/ext \
    && $SHELL ./libyuv.cmd \
    && $SHELL ./aom.cmd \
    && $SHELL ./libsharpyuv.cmd \
    && $SHELL ./libjpeg.cmd \
    && $SHELL ./zlibpng.cmd \
    && cd .. \
    && cmake -S . -B build -DBUILD_SHARED_LIBS=OFF -DAVIF_CODEC_AOM=LOCAL -DAVIF_CODEC_SVT=SYSTEM -DAVIF_LIBYUV=LOCAL -DAVIF_LIBSHARPYUV=LOCAL -DAVIF_JPEG=LOCAL -DAVIF_ZLIBPNG=LOCAL -DAVIF_BUILD_APPS=ON \
    && cmake --build build --parallel \
    && cmake --install build \
    && cd /root/local \
    && rm -rf /root/local/libavif_aom

# install pngcheck
RUN git clone https://github.com/pnggroup/pngcheck.git -b v4.0.1 \
    && cd pngcheck \
    && mkdir build && cd build \
    && cmake -DPNGCHECK_USE_SYSTEM_ZLIB=OFF .. \
    && cmake --build . \
    && cmake --install . \
    && rm -rf /root/local/pngcheck

# Install png_cicp_editor
RUN git clone https://github.com/ProgramMax/png_cicp_editor.git -b v3.0 \
    && cd png_cicp_editor \
    && cd Projects/Make && make \
    && cp ./Build/png_cicp_editor /usr/local/bin/ \
    && rm -rf /root/local/png_cicp_editor

# Install Bento4 (mp4dump)
RUN git clone https://github.com/axiomatic-systems/Bento4.git \
    && cd Bento4 \
    && mkdir cmakebuild \
    && cd cmakebuild \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make & make install \
    && cd ../../ \
    && rm -rf Bento4

# Install GPAC for mp4 container edit
RUN git clone https://github.com/gpac/gpac.git \
    && cd gpac \
    && ./configure \
    && make & make install \
    && cd ../ \
    && rm -rf gpac

# Install Noto Fonts
RUN mkdir font_install \
    && cd font_install \
    && wget https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip \
    && unzip NotoSansCJKjp-hinted.zip \
    && mkdir -p /usr/share/fonts/opentype/noto \
    && cp *otf /usr/share/fonts/opentype/noto \
    && fc-cache -f -v \
    && cd .. \
    && rm -rf font_install

# Install BUZ Fonts
RUN mkdir /home/font_install \
    && cd /home/font_install \
    && git clone https://github.com/googlefonts/morisawa-biz-ud-gothic.git \
    && cd morisawa-biz-ud-gothic/fonts/ttf/ \
    && mkdir -p /usr/share/fonts/opentype/bizudp \
    && cp *.ttf /usr/share/fonts/opentype/bizudp \
    && fc-cache -f -v \
    && rm -rf /home/font_install/morisawa-biz-ud-gothic

RUN ldconfig

# User Settings
ARG UID=1000
# GID 1000 is used for docker group
ARG GID=1001
ARG USER_NAME="devuser"

RUN groupadd -g ${GID} ${USER_NAME} \
    && useradd -m -s /bin/bash -u ${UID} -g ${GID} -p '$1$jQHQ4kCa$qx6X2s6ee1GqpGszqZNtV0' ${USER_NAME} \
    && usermod -aG sudo ${USER_NAME} \
    && chage -d 0 ${USER_NAME}

WORKDIR /work/src
USER ${USER_NAME}

ENV LANG=ja_JP.UTF-8
ENV LC_ALL=ja_JP.UTF-8

# Git init settings
RUN git config --global user.name toru.yoshihara \
    && git config --global user.email dummy_address@gmail.com \
    && git config --global core.autocrlf false

CMD ["bash"]
