#!/bin/bash

set -euo pipefail

ffmpeg_bin="/opt/my_ffmpeg_out/bin/ffmpeg"
x265_bin="x265"
video_size="3840x2160"
fps="24"
frames="1"

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

x265_csp_for_subsampling() {
    local subsampling="$1"

    case "$subsampling" in
        420)
            printf 'i420'
            ;;
        422)
            printf 'i422'
            ;;
        444)
            printf 'i444'
            ;;
        *)
            echo "Error: unsupported subsampling: $subsampling" >&2
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

mkdir -p ./raw/x265_yuv ./raw/hevc

for bit_depth in "${bit_depth_list[@]}"; do
    src_fname="$(src_fname_for_bit_depth "$bit_depth")"

    for subsampling in "${subsampling_list[@]}"; do
        ffmpeg_pix_fmt="$(ffmpeg_pix_fmt_for_params "$subsampling" "$bit_depth")"
        file_pix_fmt="$(file_pix_fmt_for_params "$subsampling" "$bit_depth")"
        x265_csp="$(x265_csp_for_subsampling "$subsampling")"

        for gamut in "${gamut_list[@]}"; do
            mtx_str="$(matrix_for_gamut "$gamut")"
            primaries_str="$(primaries_for_gamut "$gamut")"
            trc_str="bt709"
            yuv_fname="./raw/x265_yuv/src_3840x2160_${file_pix_fmt}_${gamut}.yuv"
            hevc_fname="./raw/hevc/x265_3840x2160_${file_pix_fmt}_${gamut}.hevc"
            dst_fname="./raw/x265_3840x2160_${file_pix_fmt}_${gamut}.mp4"

            rm -f -- "$yuv_fname" "$hevc_fname" "$dst_fname"

            run_cmd \
                "$ffmpeg_bin" \
                -hide_banner \
                -y \
                -i "$src_fname" \
                -vf scale=in_range=full:out_range=limited:out_color_matrix="$mtx_str" \
                -pix_fmt "$ffmpeg_pix_fmt" \
                -f rawvideo \
                "$yuv_fname"

            run_cmd \
                "$x265_bin" \
                --input "$yuv_fname" \
                --input-res "$video_size" \
                --fps "$fps" \
                --frames "$frames" \
                --input-depth "$bit_depth" \
                --input-csp "$x265_csp" \
                --range limited \
                --colorprim "$primaries_str" \
                --transfer "$trc_str" \
                --colormatrix "$mtx_str" \
                --lossless \
                --output "$hevc_fname"

            run_cmd \
                "$ffmpeg_bin" \
                -hide_banner \
                -y \
                -framerate "$fps" \
                -i "$hevc_fname" \
                -map 0:v:0 \
                -c:v copy \
                -color_primaries "$primaries_str" \
                -color_trc "$trc_str" \
                -colorspace "$mtx_str" \
                "$dst_fname"
        done
    done
done
