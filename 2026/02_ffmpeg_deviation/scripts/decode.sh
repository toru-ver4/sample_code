#!/bin/bash

set -euo pipefail

ffmpeg_bin="/opt/my_ffmpeg_out/bin/ffmpeg"
video_size="3840x2160"

bit_depth_list=(8 10 12)
gamut_list=("bt.709" "bt.2020")
subsampling_list=("420" "422" "444")

file_pix_fmt_for_params() {
    local subsampling="$1"
    local bit_depth="$2"
    printf 'yuv%sp%dle' "$subsampling" "$bit_depth"
}

ffmpeg_pix_fmt_for_params() {
    local subsampling="$1"
    local bit_depth="$2"

    if [[ "$bit_depth" == "8" ]]; then
        printf 'yuv%sp' "$subsampling"
    else
        printf 'yuv%sp%dle' "$subsampling" "$bit_depth"
    fi
}

rgb_pix_fmt_for_bit_depth() {
    local bit_depth="$1"

    case "$bit_depth" in
        8)
            printf 'rgb24'
            ;;
        10)
            printf 'gbrp10le'
            ;;
        12)
            printf 'gbrp12le'
            ;;
        *)
            echo "Error: unsupported bit depth: $bit_depth" >&2
            return 1
            ;;
    esac
}

matrix_for_gamut() {
    local gamut="$1"

    case "$gamut" in
        "bt.709")
            printf 'bt709'
            ;;
        "bt.2020")
            printf 'bt2020nc'
            ;;
        *)
            echo "Error: unsupported gamut: $gamut" >&2
            return 1
            ;;
    esac
}

mkdir -p ./img

for bit_depth in "${bit_depth_list[@]}"; do
    rgb_pix_fmt="$(rgb_pix_fmt_for_bit_depth "$bit_depth")"

    for subsampling in "${subsampling_list[@]}"; do
        ffmpeg_pix_fmt="$(ffmpeg_pix_fmt_for_params "$subsampling" "$bit_depth")"
        file_pix_fmt="$(file_pix_fmt_for_params "$subsampling" "$bit_depth")"

        for gamut in "${gamut_list[@]}"; do
            mtx_str="$(matrix_for_gamut "$gamut")"
            src_fname="./raw/ref_3840x2160_${file_pix_fmt}_${gamut}.yuv"
            dst_fname="./img/ffmpeg_decode_${file_pix_fmt}_${gamut}.dpx"

            rm -f -- "$dst_fname"

            printf '%q ' \
                "$ffmpeg_bin" \
                -hide_banner \
                -y \
                -f rawvideo \
                -pix_fmt "$ffmpeg_pix_fmt" \
                -video_size "$video_size" \
                -framerate 24 \
                -i "$src_fname" \
                -frames:v 1 \
                -vf "scale=in_range=limited:out_range=full:in_color_matrix=$mtx_str" \
                -pix_fmt "$rgb_pix_fmt" \
                "$dst_fname"
            printf '\n'

            "$ffmpeg_bin" \
                -hide_banner \
                -y \
                -f rawvideo \
                -pix_fmt "$ffmpeg_pix_fmt" \
                -video_size "$video_size" \
                -framerate 24 \
                -i "$src_fname" \
                -frames:v 1 \
                -vf scale=in_range=limited:out_range=full:in_color_matrix="$mtx_str" \
                -pix_fmt "$rgb_pix_fmt" \
                "$dst_fname"
        done
    done
done
