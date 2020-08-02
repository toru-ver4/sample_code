#!/bin/sh

# define
AUDIO_FILE="/work/src/2020/005_make_countdown_movie/wav/countdown.wav"
IN_FILE="/work/overuse/2020/005_make_countdown_movie/movie_seq/movie_SDR_1920x1080_24fps_%4d.png"
# IN_FILE="/work/overuse/2020/005_make_countdown_movie/movie_seq/movie_SDR_3840x2160_24fps_0000.png"
MP4_FILE_444="/work/overuse/2020/005_make_countdown_movie/ffmpeg_out/Countdown_1920x1080_24P_Rev05_H264_444.mp4"
MP4_FILE_422="/work/overuse/2020/005_make_countdown_movie/ffmpeg_out/Countdown_1920x1080_24P_Rev05_H264_422.mp4"
MP4_FILE_420="/work/overuse/2020/005_make_countdown_movie/ffmpeg_out/Countdown_1920x1080_24P_Rev05_H264_420.mp4"

# encode
ffmpeg -framerate 24 -i ${IN_FILE} -i ${AUDIO_FILE}      -pix_fmt yuv444p -c:a aac ${MP4_FILE_444} -y
ffmpeg -framerate 24 -i ${IN_FILE} -i ${AUDIO_FILE} -c:v libx264 -qp 0 -pix_fmt yuv422p -c:a aac ${MP4_FILE_422} -y
ffmpeg -framerate 24 -i ${IN_FILE} -i ${AUDIO_FILE} -c:v libx264 -qp 0 -pix_fmt yuv420p -c:a aac ${MP4_FILE_420} -y
