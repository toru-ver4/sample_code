#!/bin/sh

/opt/my_ffmpeg_out/bin/ffmpeg \
    -hide_banner \
    -i ./img/src_img.png \
    -vf scale=in_range=full:out_range=limited:out_color_matrix=bt709 \
    -pix_fmt yuv420p10le \
    -f rawvideo \
    ./raw/ffmpeg_yuv420p10le.yuv \
    -y
