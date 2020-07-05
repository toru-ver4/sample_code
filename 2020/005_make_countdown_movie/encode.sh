#!/bin/sh

# define
IN_FILE="/work/overuse/2020/005_make_countdown_movie/movie_seq/movie_SDR_1920x1080_24fps_0000.png"
# IN_FILE="/work/overuse/2020/005_make_countdown_movie/movie_seq/movie_SDR_3840x2160_24fps_0000.png"
MP4_FILE_444="/work/overuse/2020/005_make_countdown_movie/ffmpeg_out/444_h264.mp4"
MP4_FILE_422="/work/overuse/2020/005_make_countdown_movie/ffmpeg_out/422_h264.mp4"
MP4_FILE_420="/work/overuse/2020/005_make_countdown_movie/ffmpeg_out/420_h264.mp4"
MP4_FILE_PRORES_444="/work/overuse/2020/005_make_countdown_movie/ffmpeg_out/444_ProRes.mp4"
MP4_FILE_PRORES_422="/work/overuse/2020/005_make_countdown_movie/ffmpeg_out/422_ProRes.mp4"
STIL_FILE_444="/work/overuse/2020/005_make_countdown_movie/ffmpeg_out/444_h264_stil_%4d.png"
STIL_FILE_422="/work/overuse/2020/005_make_countdown_movie/ffmpeg_out/422_h264_stil_%4d.png"
STIL_FILE_420="/work/overuse/2020/005_make_countdown_movie/ffmpeg_out/420_h264_stil_%4d.png"
STIL_FILE_PRORES_444="/work/overuse/2020/005_make_countdown_movie/ffmpeg_out/444_ProRes_stil_%4d.png"
STIL_FILE_PRORES_422="/work/overuse/2020/005_make_countdown_movie/ffmpeg_out/422_ProRes_stil_%4d.png"

# encode
ffmpeg -loop 1 -framerate 24 -i ${IN_FILE} -t 1 -c:v libx264 -qp 0 -pix_fmt yuv444p ${MP4_FILE_444} -y
ffmpeg -loop 1 -framerate 24 -i ${IN_FILE} -t 1 -c:v libx264 -qp 0 -pix_fmt yuv422p ${MP4_FILE_422} -y
ffmpeg -loop 1 -framerate 24 -i ${IN_FILE} -t 1 -c:v libx264 -qp 0 -pix_fmt yuv420p ${MP4_FILE_420} -y
ffmpeg -loop 1 -framerate 24 -i ${IN_FILE} -t 1 -c:v prores_ks -profile:v 4 -pix_fmt yuv444p10le ${MP4_FILE_PRORES_444} -y
ffmpeg -loop 1 -framerate 24 -i ${IN_FILE} -t 1 -c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le ${MP4_FILE_PRORES_422} -y

# decode (stil image)
ffmpeg -i ${MP4_FILE_444} -vf "trim=start_frame=0:end_frame=1,setpts=PTS-STARTPTS" -vsync 0 ${STIL_FILE_444} -y
ffmpeg -i ${MP4_FILE_422} -vf "trim=start_frame=0:end_frame=1,setpts=PTS-STARTPTS" -vsync 0 ${STIL_FILE_422} -y
ffmpeg -i ${MP4_FILE_420} -vf "trim=start_frame=0:end_frame=1,setpts=PTS-STARTPTS" -vsync 0 ${STIL_FILE_420} -y
ffmpeg -i ${MP4_FILE_PRORES_444} -vf "trim=start_frame=0:end_frame=1,setpts=PTS-STARTPTS" -vsync 0 ${STIL_FILE_PRORES_444} -y
ffmpeg -i ${MP4_FILE_PRORES_422} -vf "trim=start_frame=0:end_frame=1,setpts=PTS-STARTPTS" -vsync 0 ${STIL_FILE_PRORES_422} -y
