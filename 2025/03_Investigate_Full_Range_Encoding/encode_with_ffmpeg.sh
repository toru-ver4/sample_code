#!/bin/bash

mkdir -p \
  "encode_data/FFmpeg/H265_MOV_Main10_Full" \
  "encode_data/FFmpeg/H265_MOV_Main10_Limited" \
  "encode_data/FFmpeg/H265_MP4_Main10_Full" \
  "encode_data/FFmpeg/ProRes_MOV_422HQ_Full"

# ffmpeg -loop 1 -color_primaries bt709 -color_trc bt709 -colorspace bt709 -r 24 -i ./img/src_img.png -t 1 -c:v libx265 -x265-params lossless=1:range=full -pix_fmt yuv422p10le -tag:v hvc1 -an -color_primaries bt709 -color_trc bt709 -colorspace bt709 ./encode_data/FFmpeg/H265_MOV_Main10_Full/H265_MOV_Main10_Full.mov -y
# ffmpeg -loop 1 -color_primaries bt709 -color_trc bt709 -colorspace bt709 -r 24 -i ./img/src_img.png -t 1 -c:v libx265 -x265-params lossless=1:range=limited -pix_fmt yuv422p10le -tag:v hvc1 -an -color_primaries bt709 -color_trc bt709 -colorspace bt709 ./encode_data/FFmpeg/H265_MOV_Main10_Limited/H265_MOV_Main10_Limited.mov -y
# ffmpeg -loop 1 -color_primaries bt709 -color_trc bt709 -colorspace bt709 -r 24 -i ./img/src_img.png -t 1 -c:v libx265 -x265-params lossless=1:range=full -pix_fmt yuv422p10le -tag:v hvc1 -an -color_primaries bt709 -color_trc bt709 -colorspace bt709 ./encode_data/FFmpeg/H265_MP4_Main10_Full/H265_MOV_Main10_Full.mp4 -y

ffmpeg -loop 1 -color_primaries bt709 -color_trc bt709 -colorspace bt709 -r 24 -i ./img/src_img.png -t 1 -c:v prores_ks -profile:v 3 -color_range pc                          -tag:v hvc1 -an -color_primaries bt709 -color_trc bt709 -colorspace bt709 ./encode_data/FFmpeg/ProRes_MOV_422HQ_Full/ProRes_MOV_422HQ_Full.mov -y
