# エンコードに関するメモ

## DaVinci Resolve


## FFmpeg

libaom-av1 librav1e libsvtav1 av1_nvenc av1_qsv av1_amf av1_vaapi

ffmpeg -framerate 23.976 -start_number 86400 -i "D:\abuse\Countdown\temp_seq\1920x1080_23.976P_ST2084_Rec.2020.png%08d.png" -pix_fmt yuv444p12le -c:v librav1e -lossless 1 "D:\abuse\Countdown\output.mov"

