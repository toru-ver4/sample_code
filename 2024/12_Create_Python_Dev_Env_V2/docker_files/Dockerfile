ARG PYTHON_VERSION="3.13"
ARG IMAGE_TYPE="slim"
ARG DEBIAN_VERSION="trixie"

FROM python:${PYTHON_VERSION}-${IMAGE_TYPE}-${DEBIAN_VERSION} AS builder

# Avoid clearing `DEBIAN_VERSION`
ARG DEBIAN_VERSION
ARG PYTHON_VERSION

WORKDIR /root/local

RUN rm -f /etc/apt/sources.list.d/debian.sources \
    && printf '%s\n' \
        "deb http://deb.debian.org/debian ${DEBIAN_VERSION} main" \
        "deb http://deb.debian.org/debian ${DEBIAN_VERSION}-updates main" \
        "deb http://security.debian.org/debian-security ${DEBIAN_VERSION}-security main" \
        > /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        build-essential \
        clang \
        cmake \
        dasel \
        doxygen \
        ffmpeg \
        git \
        git-core \
        imagemagick \
        libaom-dev \
        libass-dev \
        libboost-all-dev \
        libdav1d-dev \
        libx264-dev \
        libde265-dev \
        libexpat1-dev \
        libfreetype-dev \
        libfreetype6-dev \
        libgif-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libgnutls28-dev \
        libheif-dev \
        libimage-exiftool-perl \
        libimath-dev \
        libjpeg-dev \
        liblcms2-dev \
        libmp3lame-dev \
        libopencv-dev \
        libopenexr-dev \
        libopenjp2-7-dev \
        libopenvdb-dev \
        libvdpau-dev \
        libpystring-dev \
        libraw-dev \
        libsdl2-dev \
        libsvtav1enc-dev \
        libtbb-dev \
        libtiff-dev \
        libtool \
        libunistring-dev \
        libva-dev \
        libvorbis-dev \
        libvpx-dev \
        libwebp-dev \
        libwxgtk-media3.2-dev \
        libwxgtk-webview3.2-dev \
        libxcb1-dev \
        libx265-dev \
        libxcb-cursor0 \
        libxcb-shm0-dev \
        libxcb-xfixes0-dev \
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
        locales \
        meson \
        nasm \
        ninja-build \
        nlohmann-json3-dev \
        pkg-config \
        pybind11-dev \
        python3-tk \
        texinfo \
        tk-dev \
        sudo \
        unzip \
        wget \
        wx-common \
        wx3.2-headers \
        yasm \
        zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# # Install FFmpeg
# RUN mkdir -p ~/ffmpeg_sources /opt/my_ffmpeg /opt/my_ffmpeg_out

# # Insatll libfdk-aac for FFmpeg
# RUN cd ~/ffmpeg_sources \
#     && git -C fdk-aac pull 2> /dev/null || git clone --depth 1 https://github.com/mstorsjo/fdk-aac \
#     && cd fdk-aac \
#     && autoreconf -fiv \
#     && ./configure --prefix="/opt/my_ffmpeg/" --disable-shared \
#     && make -j 16 \
#     && make install \
#     && cd ~/ && rm -rf ./ffmpeg_sources/fdk-aac

# RUN cd ~/ffmpeg_sources \
#     && git -C aom pull 2> /dev/null || git clone --depth 1 https://aomedia.googlesource.com/aom \
#     && mkdir -p aom_build \
#     && cd aom_build \
#     && PATH="/opt/my_ffmpeg/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="/opt/my_ffmpeg/" -DENABLE_TESTS=OFF -DENABLE_NASM=on ../aom \
#     && PATH="/opt/my_ffmpeg/bin:$PATH" make -j 16 \
#     && make install \
#     && cd ~/ && rm -rf ./ffmpeg_sources/aom

# RUN cd ~/ffmpeg_sources \
#     && git -C SVT-AV1 pull 2> /dev/null || git clone https://gitlab.com/AOMediaCodec/SVT-AV1.git \
#     && mkdir -p SVT-AV1/build \
#     && cd SVT-AV1/build \
#     && PATH="/opt/my_ffmpeg/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="/opt/my_ffmpeg/" -DCMAKE_BUILD_TYPE=Release -DBUILD_DEC=OFF -DBUILD_SHARED_LIBS=OFF .. \
#     && PATH="/opt/my_ffmpeg/bin:$PATH" make -j16 \
#     && make install \
#     && cd ~/ && rm -rf ./ffmpeg_sources/SVT-AV1

# RUN cd ~/ffmpeg_sources \
#     && git -C dav1d pull 2> /dev/null || git clone --depth 1 https://code.videolan.org/videolan/dav1d.git \
#     && mkdir -p dav1d/build \
#     && cd dav1d/build \
#     && meson setup -Denable_tools=false -Denable_tests=false --default-library=static .. --prefix "/opt/my_ffmpeg" --libdir="/opt/my_ffmpeg/lib" \
#     && ninja -j16 \
#     && ninja install \
#     && cd ~/ && rm -rf ./ffmpeg_sources/dav1d

# RUN cd ~/'ffmpeg_sources' \
#     && git -C vmaf pull 2> /dev/null || git clone --depth 1 https://github.com/Netflix/vmaf.git \
#     && mkdir -p 'vmaf/libvmaf/build' \
#     && cd 'vmaf/libvmaf/build' \
#     && meson setup -Denable_tests=false -Denable_docs=false --buildtype=release --default-library=static '../' --prefix "/opt/my_ffmpeg/" --bindir="/opt/my_ffmpeg/bin" --libdir="/opt/my_ffmpeg/lib" \
#     && ninja -j16 \
#     && ninja install \
#     && cd ~/ && rm -rf ./ffmpeg_sources/vmaf

# RUN cd ~/ffmpeg_sources \
#     && git -C opus pull 2> /dev/null || git clone --depth 1 https://github.com/xiph/opus.git \
#     && cd opus \
#     && ./autogen.sh \
#     && ./configure --prefix="/opt/my_ffmpeg" --disable-shared \
#     && make -j16 \
#     && make install \
#     && cd ~/ && rm -rf ./ffmpeg_sources/opus

# RUN cd ~/ffmpeg_sources \
#     && git clone https://github.com/toru-ver4/FFmpeg_precise_10bit.git ffmpeg \
#     && cd ffmpeg \
#     && PATH="/opt/my_ffmpeg/bin:$PATH" PKG_CONFIG_PATH="/opt/my_ffmpeg/lib/pkgconfig" ./configure \
#         --prefix="/opt/my_ffmpeg_out" \
#         --pkg-config-flags="--static" \
#         --extra-cflags="-I/opt/my_ffmpeg/include" \
#         --extra-ldflags="-L/opt/my_ffmpeg/lib" \
#         --extra-libs="-lpthread -lm" \
#         --ld="g++" \
#         --bindir="/opt/my_ffmpeg_out/bin" \
#         --enable-gpl \
#         --enable-gnutls \
#         --enable-libaom \
#         --enable-libass \
#         --enable-libfdk-aac \
#         --enable-libfreetype \
#         --enable-libmp3lame \
#         --enable-libopus \
#         --enable-libsvtav1 \
#         --enable-libdav1d \
#         --enable-libvorbis \
#         --enable-libvpx \
#         --enable-libx264 \
#         --enable-libx265 \
#         --enable-nonfree \
#     && PATH="/opt/my_ffmpeg_out/bin:$PATH" make -j16 \
#     && make install \
#     && hash -r \
#     && cd ~/ && rm -rf ./ffmpeg_sources/ffmpeg

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
    && make -j"$(nproc)" \
    && cp ultrahdr_app /usr/local/bin/ \
    && cd / \
    && rm -rf /root/local/libultrahdr

COPY ./requirements.txt /root/local/requirements.txt

RUN pip install --upgrade pip \
        && pip install --no-cache-dir -r /root/local/requirements.txt

# Install libheif
RUN git clone https://github.com/strukturag/libheif.git -b v1.23.0   \
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
    && git clone https://github.com/AcademySoftwareFoundation/OpenColorIO.git /root/local/OpenColorIO -b v2.5.2 \
    && cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DOCIO_INSTALL_EXT_PACKAGES=MISSING -DOCIO_BUILD_PYTHON=0 -DOCIO_VERBOSE=1 /root/local/OpenColorIO \
    && make -j16 \
    && make install \
    && cd /root/local \
    && rm -rf /root/local/OpenColorIO /tmp/ociobuild

# Install OpenImageIO
RUN git clone https://github.com/OpenImageIO/oiio.git -b v3.1.14.0 \
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
    
# Install ICC XML
RUN mkdir -p /opt \
        && cd /opt \
        && git clone https://github.com/InternationalColorConsortium/iccDEV.git -b v2.3.2.1 \
        && cd iccDEV \
        && cmake --preset linux-clang -S Build/Cmake -B out/linux-clang -DENABLE_TOOLS=ON \
        && cmake --build out/linux-clang -j"$(nproc)" \
        && cmake --install out/linux-clang --prefix /opt/iccdev-install \
        && cd /opt && rm -rf iccDEV

RUN printf '%s\n' \
        /opt/iccdev-install/lib \
        > /etc/ld.so.conf.d/iccdev.conf \
    && ldconfig

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
RUN git clone https://github.com/AOMediaCodec/libavif.git -b v1.4.2 libavif_aom \
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
    && make -j16 && make install \
    && cd ../../ \
    && rm -rf Bento4

# Install GPAC for mp4 container edit
RUN git clone https://github.com/toru-ver4/gpac.git \
    && cd gpac \
    && ./configure \
    && make -j16 && make install \
    && cd ../ \
    && rm -rf gpac

# Install Noto Fonts (Noto Sans CJK JP - Japanese only)
RUN mkdir font_install \
    && cd font_install \
    && wget -O NotoSansCJKjp.zip https://github.com/googlefonts/noto-cjk/releases/download/Sans2.004/06_NotoSansCJKjp.zip \
    && unzip NotoSansCJKjp.zip \
    && mkdir -p /usr/share/fonts/opentype/noto \
    && find . -name '*.otf' -exec cp -v {} /usr/share/fonts/opentype/noto/ \; \
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

RUN sed -i 's/^# *\(ja_JP.UTF-8 UTF-8\)/\1/' /etc/locale.gen \
    && locale-gen

# User Settings
ARG UID=1000
# GID 1000 is used for docker group
ARG GID=1001
ARG USER_NAME="devuser"

RUN groupadd -g ${GID} ${USER_NAME} \
    && useradd -m -s /bin/bash -u ${UID} -g ${GID} ${USER_NAME} \
    && usermod -aG sudo ${USER_NAME} \
    && printf '%s\n' "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER_NAME} \
    && chmod 0440 /etc/sudoers.d/${USER_NAME}

WORKDIR /work/src
USER ${USER_NAME}
ENV PATH="/opt/iccdev-install/bin:${PATH}"
# ENV PATH="/opt/my_ffmpeg_out/bin:${PATH}"

ENV LANG=ja_JP.UTF-8
ENV LC_ALL=ja_JP.UTF-8

# Git init settings
RUN git config --global user.name toru.yoshihara \
    && git config --global user.email dummy_address@gmail.com \
    && git config --global core.autocrlf false

CMD ["bash"]
