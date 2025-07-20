#!/bin/bash
# avifenc ./img/BT2100-PQ.png -y 444 -d 10 --cicp 9/16/0 -r full --lossless --ignore-exif ./img/BT2100-PQ.avif
# avifenc ./img/BT2020-BT1886.png -y 444 -d 10 --cicp 1/1/0 -r full --lossless --ignore-exif ./img/BT2100-BT1886.avif

# avifenc ./img/BT2100-PQ.png --cicp 9/16/9 --ignore-exif ./img/BT2100-PQ_default.avif
# avifenc ./img/BT2020-BT1886.png --cicp 1/1/1 --ignore-exif ./img/BT2100-BT1886_default.avif

# Note:
# Using a backslash to break a string into several lines is possible,
# but inline comments within the quotes will become part of the value.
# A better approach when you need per-value comments is to use a bash array.
headroom_values=(
    0.000  # log2(203/203)
    0.979  # log2(400/203)
    1.563  # log2(600/203)
    2.300  # log2(1000/203)
    3.300  # log2(2000/203)
    4.300  # log2(4000/203)
    5.622  # log2(10000/203)
)

# combine_gain_map function created by copilot
# Usage:
#   combine_gain_map <sdr_image> <hdr_image> <output_dir> <headroom_values_array...>
#
# Parameters:
#   sdr_image       - The SDR image used as the second argument for avifgainmaputil combine.
#   hdr_image       - The HDR image used as the third argument for avifgainmaputil combine.
#   output_dir      - The directory where the output files will be saved.
#   headroom_values - A bash array containing the headroom values for the --manual-alternate-hdr-headroom option.
#
# Output File Naming:
#   The output file name is constructed as follows:
#     <sdr_image_base>-<hdr_image_base>_<headroom_value>.avif
#
#   For example, if:
#     sdr_image: "./img/BT2100-BT1886.avif"
#     hdr_image: "./img/BT2100-PQ.avif"
#     output_dir: "./hoge/"
#     headroom_value: 2.300448
#   Then the output file will be:
#     "./hoge/BT2100-BT1886-BT2100-PQ_2.300448.avif"
combine_gain_map() {
    local sdr_image="$1"    # SDR image for avifgainmaputil combine (2nd argument)
    local hdr_image="$2"    # HDR image for avifgainmaputil combine (3rd argument)
    local output_dir="$3"   # Output directory
    shift 3
    local headroom_values=( "$@" )  # Bash array of headroom values

    # Append trailing slash to output_dir if not present
    [[ "${output_dir}" != */ ]] && output_dir="${output_dir}/"

    # Extract file names (without extension) from sdr_image and hdr_image
    local sdr_filename="${sdr_image##*/}"     # e.g., BT2100-BT1886.avif
    local hdr_filename="${hdr_image##*/}"       # e.g., BT2100-PQ.avif
    local sdr_base="${sdr_filename%.*}"         # e.g., BT2100-BT1886
    local hdr_base="${hdr_filename%.*}"         # e.g., BT2100-PQ

    # Loop over each headroom value
    for headroom in "${headroom_values[@]}"; do
        # Construct the output file name according to the naming convention
        local output="${output_dir}${sdr_base}-${hdr_base}_${headroom}.avif"
        echo "Processing headroom=${headroom} -> output file: ${output}"
        avifgainmaputil combine \
            "$sdr_image" \
            "$hdr_image" \
            "$output" \
            --qgain-map 100 \
            --depth-gain-map 10 \
            --yuv-gain-map 444 \
            --cicp-base 1/1/0 \
            --cicp-alternate 9/16/0 \
            --qcolor 100 \
            --depth 10 \
            --manual-base-hdr-headroom 0.0 \
            --manual-alternate-hdr-headroom "${headroom}"
    done
}

# combine_gain_map \
#     "./src_img/1920x1080_sRGB_sRGB_0.5x.avif" \
#     "./src_img/1920x1080_ST2084_Rec.2020.avif" \
#     "./gain_map_img" \
#     "${headroom_values[@]}"

# combine_gain_map \
#     "./src_img/1920x1080_sRGB_sRGB.avif" \
#     "./src_img/1920x1080_ST2084_Rec.2020.avif" \
#     "./gain_map_img" \
#     "${headroom_values[@]}"

# combine_gain_map \
#     "./src_img/kanazawa_01_sRGB.avif" \
#     "./src_img/kanazawa_01_BT2100-PQ.avif" \
#     "./gain_map_img" \
#     "${headroom_values[@]}"

# combine_gain_map \
#     "./src_img/kenrokuen_01_sRGB.avif" \
#     "./src_img/kenrokuen_01_BT2100-PQ.avif" \
#     "./gain_map_img" \
#     "${headroom_values[@]}"

# combine_gain_map \
#     "./src_img/kanazawa_castle_01_sRGB.avif" \
#     "./src_img/kanazawa_castle_01_BT2100-PQ.avif" \
#     "./gain_map_img" \
#     "${headroom_values[@]}"

# # HDR Capacity Image (Note! Base rendition is *HDR*)
# capacity_values=(0.000 0.563 0.978 1.300 1.563 1.978 2.300 2.563 2.885 3.300 3.622 3.885 4.300 5.622 6.965)
# for capacity in "${capacity_values[@]}"; do
#     output="./gain_map_img/hdr_capacity_checker_headroom-${capacity}.avif"
#     avifgainmaputil combine \
#         ./src_img/HDR_Capacity_${capacity}_1280x720.avif \
#         ./src_img/HDR_Capacity_SDR_1280x720.avif \
#         ${output} \
#         --qgain-map 100 \
#         --depth-gain-map 10 \
#         --yuv-gain-map 444 \
#         --cicp-base 9/16/0 \
#         --cicp-alternate 9/13/0 \
#         --qcolor 100 \
#         --depth 10 \
#         --manual-base-hdr-headroom ${capacity} \
#         --manual-alternate-hdr-headroom 0.0
# done


# # blog iamge
# avifgainmaputil combine \
#     ./src_img/kenrokuen_01_sRGB.avif \
#     ./src_img/kenrokuen_01_BT2100-PQ_1000nit.avif \
#     ./gain_map_img/kenrokuen_01_sRGB_BT2100-PQ_1000nit.avif \
#     --qgain-map 80 \
#     --depth-gain-map 10 \
#     --yuv-gain-map 444 \
#     --cicp-base 1/13/0 \
#     --cicp-alternate 9/16/0 \
#     --qcolor 80 \
#     --depth 10

# avifgainmaputil combine \
#     ./src_img/kenrokuen_01_sRGB.avif \
#     ./src_img/kenrokuen_01_BT2100-PQ_10000nit.avif \
#     ./gain_map_img/kenrokuen_01_sRGB_BT2100-PQ_10000nit.avif \
#     --qgain-map 80 \
#     --depth-gain-map 10 \
#     --yuv-gain-map 444 \
#     --cicp-base 1/13/0 \
#     --cicp-alternate 9/16/0 \
#     --qcolor 80 \
#     --depth 10

# avifgainmaputil combine \
#     ./src_img/kanazawa_castle_01_sRGB_2K.avif \
#     ./src_img/kanazawa_castle_01_BT2100-PQ_2K.avif \
#     ./gain_map_img/kanazawa_castle_01_sRGB_2K-kanazawa_castle_01_BT2100-PQ_2K.avif \
#     --qgain-map 80 \
#     --depth-gain-map 10 \
#     --yuv-gain-map 444 \
#     --cicp-base 1/13/0 \
#     --cicp-alternate 9/16/0 \
#     --qcolor 80 \
#     --depth 10 \
#     --max-headroom 2.300

# avifgainmaputil combine \
#     ./src_img/1920x1080_ST2084_Rec.2020.avif \
#     ./src_img/1920x1080_sRGB_sRGB_0.5x.avif \
#     ./gain_map_img/1920x1080_ST2084_Rec.2020-1920x1080_sRGB_sRGB_0.5x.avif \
#     --qgain-map 100 \
#     --depth-gain-map 10 \
#     --yuv-gain-map 444 \
#     --cicp-base 9/16/0 \
#     --cicp-alternate 1/13/0 \
#     --qcolor 100 \
#     --depth 10 \
#     --manual-base-hdr-headroom 2.3 \
#     --manual-alternate-hdr-headroom 0.0

# avifgainmaputil combine \
#     ./src_img/kenrokuen_01_sRGB.avif \
#     ./src_img/kenrokuen_01_BT2100-PQ_1000nit.avif \
#     ./gain_map_img/kenrokuen_01_sRGB-kkenrokuen_01_BT2100-PQ_1000nit.avif \
#     --qgain-map 80 \
#     --depth-gain-map 10 \
#     --yuv-gain-map 444 \
#     --cicp-base 1/13/0 \
#     --cicp-alternate 9/16/0 \
#     --qcolor 80 \
#     --depth 10

# avifgainmaputil combine \
#     ./src_img/kenrokuen_01_sRGB.avif \
#     ./src_img/kenrokuen_01_BT2100-PQ_10000nit.avif \
#     ./gain_map_img/kenrokuen_01_sRGB-kenrokuen_01_BT2100-PQ_10000nit.avif \
#     --qgain-map 80 \
#     --depth-gain-map 10 \
#     --yuv-gain-map 444 \
#     --cicp-base 1/13/0 \
#     --cicp-alternate 9/16/0 \
#     --qcolor 80 \
#     --depth 10

# avifgainmaputil combine \
#     ./src_img/1920x1080_sRGB_sRGB.avif \
#     ./src_img/1920x1080_ST2084_Rec.2020.avif \
#     ./gain_map_img/1920x1080_sRGB_sRGB-1920x1080_ST2084_Rec.2020.avif \
#     --qgain-map 80 \
#     --depth-gain-map 10 \
#     --yuv-gain-map 444 \
#     --cicp-base 1/13/0 \
#     --cicp-alternate 9/16/0 \
#     --qcolor 80 \
#     --depth 10

# avifgainmaputil combine \
#     ./src_img/1920x1080_sRGB_sRGB_0.5x.avif \
#     ./src_img/1920x1080_ST2084_Rec.2020.avif \
#     ./gain_map_img/1920x1080_sRGB_sRGB_0.5x_0.5x-1920x1080_ST2084_Rec.2020.avif \
#     --qgain-map 80 \
#     --depth-gain-map 10 \
#     --yuv-gain-map 444 \
#     --cicp-base 1/13/0 \
#     --cicp-alternate 9/16/0 \
#     --qcolor 80 \
#     --depth 10

# avifgainmaputil combine \
#     ./src_img/HDR_Capacity_0.563_1280x720.avif \
#     ./src_img/HDR_Capacity_SDR_1280x720.avif \
#     ./gain_map_img/HDR_Capacity_0.563_1280x720_base_HDR.avif \
#     --qgain-map 100 \
#     --depth-gain-map 10 \
#     --yuv-gain-map 444 \
#     --cicp-base 9/16/0 \
#     --cicp-alternate 9/13/0 \
#     --qcolor 100 \
#     --depth 10 \
#     --manual-base-hdr-headroom 0.563 \
#     --manual-alternate-hdr-headroom 0.0

# avifgainmaputil combine \
#     ./src_img/HDR_Capacity_SDR_1280x720.avif \
#     ./src_img/HDR_Capacity_0.563_1280x720.avif \
#     ./gain_map_img/HDR_Capacity_0.563_1280x720_base_SDR.avif \
#     --qgain-map 100 \
#     --depth-gain-map 10 \
#     --yuv-gain-map 444 \
#     --cicp-base 9/13/0 \
#     --cicp-alternate 9/16/0 \
#     --qcolor 100 \
#     --depth 10 \
#     --manual-base-hdr-headroom 0.0 \
#     --manual-alternate-hdr-headroom 0.563

avifgainmaputil combine \
    ./src_img/is_SDR.avif \
    ./src_img/is_HDR.avif \
    ./gain_map_img/is_SDR-is_HDR.avif \
    --qgain-map 100 \
    --depth-gain-map 10 \
    --yuv-gain-map 444 \
    --cicp-base 1/1/0 \
    --cicp-alternate 1/16/0 \
    --qcolor 100 \
    --depth 10 \
    --manual-base-hdr-headroom 0.0 \
    --manual-alternate-hdr-headroom 0.563
