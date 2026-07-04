# Memo

## FFmpeg

v8.1 を使用。

https://www.gyan.dev/ffmpeg/builds/

## yuv format

https://chromium.googlesource.com/libyuv/libyuv/+/HEAD/docs/formats.md が参考になる

## x265

Encode

```powershell
x265cli.exe --input .\raw\src_1920x1080_I010.yuv --input-res 1920x1080 --fps 24 --frames 120 --input-depth 10 --input-csp i420 --profile main10 --lossless --output .\raw\output_1920x1080_main10.hevc
```

Decode

```powershell
ffmpeg -i output_main10_lossless.hevc -pix_fmt yuv420p10le -f rawvideo output_decoded_i010.yuv
```
