ARG PYTHON_VERSION="3.14"
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
        autoconf \
        automake \
        build-essential \
        clang \
        cmake \
        dasel \
        doxygen \
        git \
        git-core \
        imagemagick \
        libaom-dev \
        libass-dev \
        libboost-all-dev \
        libdav1d-dev \
        libx264-dev \
        libde265-dev \
        libde265-0 \
        libde265-examples \
        libexpat1-dev \
        libfreetype-dev \
        libfreetype6-dev \
        libgif-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libgnutls28-dev \
        libheif-dev \
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
        pkg-config \
        pybind11-dev \
        python3-tk \
        texinfo \
        tk-dev \
        sudo \
        unzip \
        wget \
        x265 \
        yasm \
        zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Install FFmpeg
RUN mkdir -p ~/ffmpeg_sources /opt/my_ffmpeg /opt/my_ffmpeg_out

# Insatll libfdk-aac for FFmpeg
RUN cd ~/ffmpeg_sources \
    && git -C fdk-aac pull 2> /dev/null || git clone --depth 1 https://github.com/mstorsjo/fdk-aac \
    && cd fdk-aac \
    && autoreconf -fiv \
    && ./configure --prefix="/opt/my_ffmpeg/" --disable-shared \
    && make -j 16 \
    && make install \
    && cd ~/ && rm -rf ./ffmpeg_sources/fdk-aac

RUN cd ~/ffmpeg_sources \
    && git -C aom pull 2> /dev/null || git clone --depth 1 https://aomedia.googlesource.com/aom \
    && mkdir -p aom_build \
    && cd aom_build \
    && PATH="/opt/my_ffmpeg/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="/opt/my_ffmpeg/" -DENABLE_TESTS=OFF -DENABLE_NASM=on ../aom \
    && PATH="/opt/my_ffmpeg/bin:$PATH" make -j 16 \
    && make install \
    && cd ~/ && rm -rf ./ffmpeg_sources/aom

RUN cd ~/ffmpeg_sources \
    && git -C SVT-AV1 pull 2> /dev/null || git clone https://gitlab.com/AOMediaCodec/SVT-AV1.git \
    && mkdir -p SVT-AV1/build \
    && cd SVT-AV1/build \
    && PATH="/opt/my_ffmpeg/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="/opt/my_ffmpeg/" -DCMAKE_BUILD_TYPE=Release -DBUILD_DEC=OFF -DBUILD_SHARED_LIBS=OFF .. \
    && PATH="/opt/my_ffmpeg/bin:$PATH" make -j16 \
    && make install \
    && cd ~/ && rm -rf ./ffmpeg_sources/SVT-AV1

RUN cd ~/ffmpeg_sources \
    && git -C dav1d pull 2> /dev/null || git clone --depth 1 https://code.videolan.org/videolan/dav1d.git \
    && mkdir -p dav1d/build \
    && cd dav1d/build \
    && meson setup -Denable_tools=false -Denable_tests=false --default-library=static .. --prefix "/opt/my_ffmpeg" --libdir="/opt/my_ffmpeg/lib" \
    && ninja -j16 \
    && ninja install \
    && cd ~/ && rm -rf ./ffmpeg_sources/dav1d

RUN cd ~/'ffmpeg_sources' \
    && git -C vmaf pull 2> /dev/null || git clone --depth 1 https://github.com/Netflix/vmaf.git \
    && mkdir -p 'vmaf/libvmaf/build' \
    && cd 'vmaf/libvmaf/build' \
    && meson setup -Denable_tests=false -Denable_docs=false --buildtype=release --default-library=static '../' --prefix "/opt/my_ffmpeg/" --bindir="/opt/my_ffmpeg/bin" --libdir="/opt/my_ffmpeg/lib" \
    && ninja -j16 \
    && ninja install \
    && cd ~/ && rm -rf ./ffmpeg_sources/vmaf

RUN cd ~/ffmpeg_sources \
    && git -C opus pull 2> /dev/null || git clone --depth 1 https://github.com/xiph/opus.git \
    && cd opus \
    && ./autogen.sh \
    && ./configure --prefix="/opt/my_ffmpeg" --disable-shared \
    && make -j16 \
    && make install \
    && cd ~/ && rm -rf ./ffmpeg_sources/opus

# RUN cd ~/ffmpeg_sources \
#     && git clone https://github.com/FFmpeg/FFmpeg.git -b release/8.1 ffmpeg \
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
#     && hash -r

RUN ldconfig

RUN sed -i 's/^# *\(ja_JP.UTF-8 UTF-8\)/\1/' /etc/locale.gen \
    && locale-gen

# # User Settings
# ARG UID=1000
# # GID 1000 is used for docker group
# ARG GID=1001
# ARG USER_NAME="devuser"

# RUN groupadd -g ${GID} ${USER_NAME} \
#     && useradd -m -s /bin/bash -u ${UID} -g ${GID} -p '$1$jQHQ4kCa$qx6X2s6ee1GqpGszqZNtV0' ${USER_NAME} \
#     && usermod -aG sudo ${USER_NAME} \
#     && chage -d 0 ${USER_NAME}

WORKDIR /work/src
# USER ${USER_NAME}
# ENV PATH="/opt/my_ffmpeg_out/bin:${PATH}"

ENV LANG=ja_JP.UTF-8
ENV LC_ALL=ja_JP.UTF-8

# # Git init settings
# RUN git config --global user.name toru.yoshihara \
#     && git config --global user.email dummy_address@gmail.com \
#     && git config --global core.autocrlf false

CMD ["bash"]
