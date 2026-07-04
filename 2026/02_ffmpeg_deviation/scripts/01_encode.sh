#!/bin/bash

set -euo pipefail

ffmpeg_bin="/opt/my_ffmpeg_out/bin/ffmpeg"

bit_depth_list=(8 10 12)
gamut_list=("bt.709" "bt.2020")
subsampling_list=("420" "422" "444")

src_fname_for_bit_depth() {
    local bit_depth="$1"
    printf './img/src_img_v2_%02d-bit.dpx' "$bit_depth"
}

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

mkdir -p ./raw

for bit_depth in "${bit_depth_list[@]}"; do
    src_fname="$(src_fname_for_bit_depth "$bit_depth")"

    for subsampling in "${subsampling_list[@]}"; do
        ffmpeg_pix_fmt="$(ffmpeg_pix_fmt_for_params "$subsampling" "$bit_depth")"
        file_pix_fmt="$(file_pix_fmt_for_params "$subsampling" "$bit_depth")"

        for gamut in "${gamut_list[@]}"; do
            mtx_str="$(matrix_for_gamut "$gamut")"
            dst_fname="./raw/ffmpeg_3840x2160_${file_pix_fmt}_${gamut}.yuv"

            rm -f -- "$dst_fname"

            printf '%q ' \
                "$ffmpeg_bin" \
                -hide_banner \
                -y \
                -i "$src_fname" \
                -vf "scale=in_range=full:out_range=limited:out_color_matrix=$mtx_str" \
                -pix_fmt "$ffmpeg_pix_fmt" \
                -f rawvideo \
                "$dst_fname"
            printf '\n'

            "$ffmpeg_bin" \
                -hide_banner \
                -y \
                -i "$src_fname" \
                -vf scale=in_range=full:out_range=limited:out_color_matrix="$mtx_str" \
                -pix_fmt "$ffmpeg_pix_fmt" \
                -f rawvideo \
                "$dst_fname"
        done
    done
done
