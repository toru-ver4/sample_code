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

# Install lubultrahdr
RUN git clone https://github.com/google/libultrahdr.git \
    && cd libultrahdr \
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

# Install OpenImageIO
RUN git clone https://github.com/OpenImageIO/oiio.git -b v2.5.14.0 \
        && cd oiio \
        && mkdir build \
        && cd build \
        && cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DPYTHON_VERSION=${PYTHON_VERSION} -DUSE_PYTHON=1 -DVERBOSE=1 -DSTOP_ON_WARNING=0 .. \
        && make install -j16 \
        && cd /root/local \
        && rm -rf /root/local/oiio
    
# Install OpencolorIO
RUN mkdir /tmp/ociobuild \
        && cd /tmp/ociobuild \
        && git clone https://github.com/AcademySoftwareFoundation/OpenColorIO.git /root/local/OpenColorIO -b v2.3.2 \
        && cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DOCIO_INSTALL_EXT_PACKAGES=MISSING -DOCIO_BUILD_PYTHON=0 -DOCIO_VERBOSE=1 /root/local/OpenColorIO \
        && make -j16 \
        && make install \
        && cd /root/local \
        && rm -rf /root/local/OpenColorIO /tmp/ociobuild

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
RUN git clone https://github.com/AOMediaCodec/libavif.git -b v1.0.3 libavif_aom \
    && cd libavif_aom/ext \
    && $SHELL ./libyuv.cmd \
    && $SHELL ./aom.cmd \
    && $SHELL ./libsharpyuv.cmd \
    && $SHELL ./libjpeg.cmd \
    && $SHELL ./zlibpng.cmd \
    && cd .. \
    && cmake -S . -B build -DBUILD_SHARED_LIBS=OFF -DAVIF_CODEC_AOM=LOCAL -DAVIF_LIBYUV=LOCAL -DAVIF_LIBSHARPYUV=LOCAL -DAVIF_JPEG=LOCAL -DAVIF_ZLIBPNG=LOCAL -DAVIF_BUILD_APPS=ON \
    && cmake --build build --parallel \
    && cmake --install build \
    && cd /root/local \
    && rm -rf /root/local/libavif_aom

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

# Git init settings
RUN git config --global user.name toru.yoshihara \
    && git config --global user.email dummy_address@gmail.com \
    && git config --global core.autocrlf true

RUN ldconfig

CMD ["bash"]
