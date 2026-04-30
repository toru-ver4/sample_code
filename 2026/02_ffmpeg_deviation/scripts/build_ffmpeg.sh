#!/bin/sh

set -eu

cd /work/src/2026/02_ffmpeg_deviation/ffmpeg_8.1_src || exit 1
PATH="/opt/my_ffmpeg/bin:$PATH" PKG_CONFIG_PATH="/opt/my_ffmpeg/lib/pkgconfig" ./configure \
    --prefix="/opt/my_ffmpeg_out" \
    --pkg-config-flags="--static" \
    --extra-cflags="-I/opt/my_ffmpeg/include" \
    --extra-ldflags="-L/opt/my_ffmpeg/lib" \
    --extra-libs="-lpthread -lm" \
    --ld="g++" \
    --bindir="/opt/my_ffmpeg_out/bin" \
    --enable-gpl \
    --enable-gnutls \
    --enable-libaom \
    --enable-libass \
    --enable-libfdk-aac \
    --enable-libfreetype \
    --enable-libmp3lame \
    --enable-libopus \
    --enable-libsvtav1 \
    --enable-libdav1d \
    --enable-libvorbis \
    --enable-libvpx \
    --enable-libx264 \
    --enable-libx265 \
    --enable-nonfree
PATH="/opt/my_ffmpeg_out/bin:$PATH" make -j12
make install
hash -r
