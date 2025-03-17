# PowerShell スクリプト例

# 作成するディレクトリの一覧
$directories = @(
    "encode_data\FFmpeg\H265_MOV_Main10_Full",
    "encode_data\FFmpeg\H265_MOV_Main10_Limited",
    "encode_data\FFmpeg\H265_MP4_Main10_Full",
    "encode_data\FFmpeg\H265_NVENC_MOV_Main10_Full",
    "encode_data\FFmpeg\H265_NVENC_MP4_Main10_Full",
    "encode_data\FFmpeg\ProRes_MOV_422HQ_Full",
    "encode_data\FFmpeg\DNxHR_MOV_HQX_10-bit_Full"
)

# 各ディレクトリが存在しない場合は作成
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
}

# ffmpeg コマンドの実行
ffmpeg -loop 1 -color_primaries bt709 -color_trc bt709 -colorspace bt709 -r 24 -i ".\img\src_img.dpx" -t 1 `
    -c:v libx265 -x265-params "lossless=1:range=full:colorprim=1:transfer=1:colormatrix=1" -pix_fmt yuv422p10le -tag:v hvc1 -an `
    -color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range pc "encode_data\FFmpeg\H265_MOV_Main10_Full\H265_MOV_Main10_Full.mov" -y

ffmpeg -loop 1 -color_primaries bt709 -color_trc bt709 -colorspace bt709 -r 24 -i ".\img\src_img.dpx" -t 1 `
    -c:v libx265 -x265-params "lossless=1:range=limited:colorprim=1:transfer=1:colormatrix=1" -pix_fmt yuv422p10le -tag:v hvc1 -an `
    -color_primaries bt709 -color_trc bt709 -colorspace bt709 "encode_data\FFmpeg\H265_MOV_Main10_Limited\H265_MOV_Main10_Limited.mov" -y

ffmpeg -loop 1 -color_primaries bt709 -color_trc bt709 -colorspace bt709 -r 24 -i ".\img\src_img.dpx" -t 1 `
    -c:v libx265 -x265-params "lossless=1:range=full:colorprim=1:transfer=1:colormatrix=1" -pix_fmt yuv422p10le -tag:v hvc1 -an `
    -color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range pc "encode_data\FFmpeg\H265_MP4_Main10_Full\H265_MP4_Main10_Full.mp4" -y

ffmpeg -loop 1 -color_primaries bt709 -color_trc bt709 -colorspace bt709 -r 24 -i ".\img\src_img.dpx" -t 1 `
    -c:v hevc_nvenc -preset slow -qp 0 -profile:v 1 -pix_fmt p010le -tag:v hvc1 -an `
    -color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range pc "encode_data\FFmpeg\H265_NVENC_MOV_Main10_Full\H265_NVENC_MOV_Main10_Full.mov" -y

ffmpeg -loop 1 -color_primaries bt709 -color_trc bt709 -colorspace bt709 -r 24 -i ".\img\src_img.dpx" -t 1 `
    -c:v hevc_nvenc -preset slow -qp 0 -profile:v 1 -pix_fmt p010le -tag:v hvc1 -an `
    -color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range pc "encode_data\FFmpeg\H265_NVENC_MP4_Main10_Full\H265_NVENC_MP4_Main10_Full.mp4" -y

ffmpeg -loop 1 -color_primaries bt709 -color_trc bt709 -colorspace bt709 -r 24 -i ".\img\src_img.dpx" -t 1 `
    -c:v prores_ks -profile:v 3 -color_range pc -tag:v hvc1 -an `
    -color_primaries bt709 -color_trc bt709 -colorspace bt709 "encode_data\FFmpeg\ProRes_MOV_422HQ_Full\ProRes_MOV_422HQ_Full.mov" -y

ffmpeg -loop 1 -color_primaries bt709 -color_trc bt709 -colorspace bt709 -r 24 -i ".\img\src_img.dpx" -t 1 `
-c:v dnxhd -pix_fmt yuv422p10le -profile:v 4 -tag:v hvc1 -an `
    -color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range pc "encode_data\FFmpeg\DNxHR_MOV_HQX_10-bit_Full\DNxHR_MOV_HQX_10-bit_Full.mov" -y
