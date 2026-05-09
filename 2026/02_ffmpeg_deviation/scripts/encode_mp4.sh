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

primaries_for_gamut() {
    local gamut="$1"

    case "$gamut" in
        "bt.709")
            printf 'bt709'
            ;;
        "bt.2020")
            printf 'bt2020'
            ;;
        *)
            echo "Error: unsupported gamut: $gamut" >&2
            return 1
            ;;
    esac
}

run_cmd() {
    printf '%q ' "$@"
    printf '\n'

    "$@"
}

mkdir -p ./raw

for bit_depth in "${bit_depth_list[@]}"; do
    src_fname="$(src_fname_for_bit_depth "$bit_depth")"

    for subsampling in "${subsampling_list[@]}"; do
        ffmpeg_pix_fmt="$(ffmpeg_pix_fmt_for_params "$subsampling" "$bit_depth")"
        file_pix_fmt="$(file_pix_fmt_for_params "$subsampling" "$bit_depth")"

        for gamut in "${gamut_list[@]}"; do
            mtx_str="$(matrix_for_gamut "$gamut")"
            primaries_str="$(primaries_for_gamut "$gamut")"
            trc_str="bt709"
            dst_fname="./raw/ffmpeg_3840x2160_${file_pix_fmt}_${gamut}.mp4"

            rm -f -- "$dst_fname"

            run_cmd \
                "$ffmpeg_bin" \
                -hide_banner \
                -y \
                -color_primaries "$primaries_str" \
                -color_trc "$trc_str" \
                -colorspace "$mtx_str" \
                -i "$src_fname" \
                -c:v libx265 \
                -x265-params lossless=1 \
                -pix_fmt "$ffmpeg_pix_fmt" \
                -color_primaries "$primaries_str" \
                -color_trc "$trc_str" \
                -colorspace "$mtx_str" \
                "$dst_fname"
        done
    done
done
