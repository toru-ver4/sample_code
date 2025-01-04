#!/bin/sh

ffmpeg -f lavfi -i "color=black:s=1280x720:r=60" -t 10 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_1280x720_60P.mp4
ffmpeg -f lavfi -i "color=black:s=1280x720:r=50" -t 10 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_1280x720_50P.mp4
ffmpeg -f lavfi -i "color=black:s=1280x720:r=30" -t 10 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_1280x720_30P.mp4
ffmpeg -f lavfi -i "color=black:s=1280x720:r=25" -t 10 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_1280x720_25P.mp4
ffmpeg -f lavfi -i "color=black:s=1280x720:r=24" -t 10 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_1280x720_24P.mp4

ffmpeg -f lavfi -i "color=black:s=1920x1080:r=60" -t 10 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_1920x1080_60P.mp4
ffmpeg -f lavfi -i "color=black:s=1920x1080:r=50" -t 10 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_1920x1080_50P.mp4
ffmpeg -f lavfi -i "color=black:s=1920x1080:r=30" -t 10 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_1920x1080_30P.mp4
ffmpeg -f lavfi -i "color=black:s=1920x1080:r=25" -t 10 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_1920x1080_25P.mp4
ffmpeg -f lavfi -i "color=black:s=1920x1080:r=24" -t 10 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_1920x1080_24P.mp4

ffmpeg -f lavfi -i "color=black:s=2560x1440:r=60" -t 10 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_2560x1440_60P.mp4
ffmpeg -f lavfi -i "color=black:s=2560x1440:r=50" -t 10 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_2560x1440_50P.mp4
ffmpeg -f lavfi -i "color=black:s=2560x1440:r=30" -t 10 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_2560x1440_30P.mp4
ffmpeg -f lavfi -i "color=black:s=2560x1440:r=25" -t 10 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_2560x1440_25P.mp4
ffmpeg -f lavfi -i "color=black:s=2560x1440:r=24" -t 10 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_2560x1440_24P.mp4

ffmpeg -f lavfi -i "color=black:s=3840x2160:r=60" -t 10 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_3840x2160_60P.mp4
ffmpeg -f lavfi -i "color=black:s=3840x2160:r=50" -t 10 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_3840x2160_50P.mp4
ffmpeg -f lavfi -i "color=black:s=3840x2160:r=30" -t 10 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_3840x2160_30P.mp4
ffmpeg -f lavfi -i "color=black:s=3840x2160:r=25" -t 10 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_3840x2160_25P.mp4
ffmpeg -f lavfi -i "color=black:s=3840x2160:r=24" -t 10 -c:v libx265 -x265-params "qpmin=51:deblock=0:sao=0:min-cu-size=32:wpp=0:bframes=0:ref=1:info=0" -pix_fmt yuv420p ./videos/dummy_video_3840x2160_24P.mp4
