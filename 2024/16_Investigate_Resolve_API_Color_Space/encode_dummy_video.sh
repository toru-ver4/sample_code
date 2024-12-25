#!/bin/sh

ffmpeg -f lavfi -i "color=black:s=1280x720:r=60" -t 60 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_60P.mp4
ffmpeg -f lavfi -i "color=black:s=1280x720:r=50" -t 60 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_50P.mp4
ffmpeg -f lavfi -i "color=black:s=1280x720:r=30" -t 60 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_30P.mp4
ffmpeg -f lavfi -i "color=black:s=1280x720:r=25" -t 60 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_25P.mp4
ffmpeg -f lavfi -i "color=black:s=1280x720:r=24" -t 60 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_24P.mp4
