#!/bin/bash

set -euo pipefail

src_fname_list=(
    "./img/src_img_v2_08-bit.dpx"
    "./img/src_img_v2_08-bit.dpx"
    "./img/src_img_v2_10-bit.dpx"
    "./img/src_img_v2_10-bit.dpx"
    "./img/src_img_v2_12-bit.dpx"
    "./img/src_img_v2_12-bit.dpx"
)

dst_fname_list=(
    "./raw/ffmpeg_1920x1080_yuv420p8le_bt.709.yuv"
    "./raw/ffmpeg_1920x1080_yuv420p8le_bt.2020.yuv"
    "./raw/ffmpeg_1920x1080_yuv420p10le_bt.709.yuv"
    "./raw/ffmpeg_1920x1080_yuv420p10le_bt.2020.yuv"
    "./raw/ffmpeg_1920x1080_yuv420p12le_bt.709.yuv"
    "./raw/ffmpeg_1920x1080_yuv420p12le_bt.2020.yuv"
)

mtx_str_list=(
    "bt709"
    "bt2020nc"
    "bt709"
    "bt2020nc"
    "bt709"
    "bt2020nc"
)

pix_fmt_list=(
    "yuv420p"
    "yuv420p"
    "yuv420p10le"
    "yuv420p10le"
    "yuv420p12le"
    "yuv420p12le"
)

if [[ ${#src_fname_list[@]} -ne ${#dst_fname_list[@]} ]] \
    || [[ ${#src_fname_list[@]} -ne ${#mtx_str_list[@]} ]] \
    || [[ ${#src_fname_list[@]} -ne ${#pix_fmt_list[@]} ]]; then
    echo "Error: encode parameter list lengths do not match." >&2
    exit 1
fi

for ((i=0; i<${#src_fname_list[@]}; i++)); do
    dst_fname="${dst_fname_list[i]}"
    rm -f -- "$dst_fname"
done

for ((i=0; i<${#src_fname_list[@]}; i++)); do
    src_fname="${src_fname_list[i]}"
    dst_fname="${dst_fname_list[i]}"
    mtx_str="${mtx_str_list[i]}"
    pix_fmt="${pix_fmt_list[i]}"

    printf '%q ' \
        /opt/my_ffmpeg_out/bin/ffmpeg \
        -hide_banner \
        -y \
        -i "$src_fname" \
        -vf "scale=in_range=full:out_range=limited:out_color_matrix=$mtx_str" \
        -pix_fmt "$pix_fmt" \
        -f rawvideo \
        "$dst_fname"
    printf '\n'

    /opt/my_ffmpeg_out/bin/ffmpeg \
        -hide_banner \
        -y \
        -i "$src_fname" \
        -vf scale=in_range=full:out_range=limited:out_color_matrix="$mtx_str" \
        -pix_fmt "$pix_fmt" \
        -f rawvideo \
        "$dst_fname"
done
