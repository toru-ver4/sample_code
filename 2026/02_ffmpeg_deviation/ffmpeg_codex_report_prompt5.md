# FFmpeg precise 10-bit 誤差修正レポート

## 修正したファイル

- `ffmpeg_precise_10bit/libswscale/input.c`
- `ffmpeg_precise_10bit/libswscale/output.c`
- `ffmpeg_precise_10bit/libswscale/swscale.c`
- `ffmpeg_precise_10bit/libswscale/x86/swscale.c`

## 修正内容

### encode: RGB full range -> YUV limited range

`libswscale/input.c` の `planar_rgb16_s*_to_y()` / `planar_rgb16_s*_to_uv()` を修正した。

従来は N-bit RGB 入力を実質 `2^N` 基準の shift で内部 14-bit YUV へ落としていたため、10-bit/12-bit の白側で過大誤差が出ていた。変更後は RGB 係数和 `sum` に対して以下の形で丸める。

```text
offset + round(sum * 255 / (((2^N) - 1) * 2^(RGB2YUV_SHIFT - 6)))
```

`sum < 0` の chroma も対称に丸める。

### decode: YUV limited range -> RGB full range

`libswscale/output.c` の `yuv2gbrp_full_X_c()` で、10-bit/12-bit などの planar GBRP 出力時に、従来の `255 << (N - 8)` 基準の full-range 値を `2^N - 1` 基準へ再スケールする処理を追加した。

```text
round(value * ((2^N) - 1) / (255 << (N - 8)))
```

12-bit については残った丸めバイアスを抑えるため、分母を 1 小さくし、12-bit 専用の丸め項を使っている。これにより 12-bit decode の gray diff は最大 1 に収まった。

### range convert constants

`libswscale/swscale.c` の limited-range 定数を shift ではなく N-bit full scale からの丸めに変更した。

```text
mpeg_min     = round( 16 * ((2^N) - 1) / 255)
mpeg_max_lum = round(235 * ((2^N) - 1) / 255)
mpeg_max_chr = round(240 * ((2^N) - 1) / 255)
```

### x86 SIMD override

x86 ASM は修正しない方針のため、`libswscale/x86/swscale.c` で以下の override を無効化し、修正済み C 実装へフォールバックさせた。

- RGB -> YUV: `GBRP10` / `GBRAP10` / `GBRP12` / `GBRAP12` の SSE2/SSE4/AVX2 入力 override
- YUV -> RGB: `GBRP10` / `GBRAP10` / `GBRP12` / `GBRAP12` の SSE2/SSE4/AVX2 `yuv2gbrp` 出力 override

理由は、既存 x86 ASM が shift ベースの 8-bit 由来スケーリングを含み、C 側の `2^N - 1` 丸めと一致しないため。

## 検証

実行したコマンド:

```bash
docker run -i -P --name ffmpeg_investigation \
  -v /mnt/c/Users/toruv/OneDrive/work/sample_code:/work/src \
  --rm takuver4/ffmpeg_investigation:rev02 \
  bash -lc 'cd /work/src/2026/02_ffmpeg_deviation && ./scripts/build_ffmpeg.sh && ./scripts/encode.sh && ./scripts/decode.sh'

source /mnt/c/Users/toruv/OneDrive/work/sample_code/.venv_wsl/bin/activate
python3 ./scripts/check_10bit_diff.py
```

最終評価:

```text
[NG] encode  8-bit yuv420p8le  bt.709  gray_diff = 1, color_diff = 3
[NG] decode  8-bit yuv420p8le  bt.709  gray_diff = 1, color_diff = 3
[NG] encode  8-bit yuv420p8le  bt.2020 gray_diff = 1, color_diff = 3
[NG] decode  8-bit yuv420p8le  bt.2020 gray_diff = 1, color_diff = 3
[OK] encode  8-bit yuv422p8le  bt.709  gray_diff = 1, color_diff = 2
[NG] decode  8-bit yuv422p8le  bt.709  gray_diff = 1, color_diff = 3
[OK] encode  8-bit yuv422p8le  bt.2020 gray_diff = 1, color_diff = 2
[NG] decode  8-bit yuv422p8le  bt.2020 gray_diff = 1, color_diff = 3
[OK] encode  8-bit yuv444p8le  bt.709  gray_diff = 1, color_diff = 2
[OK] decode  8-bit yuv444p8le  bt.709  gray_diff = 1, color_diff = 2
[OK] encode  8-bit yuv444p8le  bt.2020 gray_diff = 1, color_diff = 2
[OK] decode  8-bit yuv444p8le  bt.2020 gray_diff = 1, color_diff = 2
[OK] encode 10-bit yuv420p10le bt.709  gray_diff = 1, color_diff = 2
[OK] decode 10-bit yuv420p10le bt.709  gray_diff = 1, color_diff = 2
[OK] encode 10-bit yuv420p10le bt.2020 gray_diff = 1, color_diff = 2
[OK] decode 10-bit yuv420p10le bt.2020 gray_diff = 1, color_diff = 2
[OK] encode 10-bit yuv422p10le bt.709  gray_diff = 1, color_diff = 2
[OK] decode 10-bit yuv422p10le bt.709  gray_diff = 1, color_diff = 2
[OK] encode 10-bit yuv422p10le bt.2020 gray_diff = 1, color_diff = 2
[OK] decode 10-bit yuv422p10le bt.2020 gray_diff = 1, color_diff = 2
[OK] encode 10-bit yuv444p10le bt.709  gray_diff = 1, color_diff = 2
[OK] decode 10-bit yuv444p10le bt.709  gray_diff = 1, color_diff = 2
[OK] encode 10-bit yuv444p10le bt.2020 gray_diff = 1, color_diff = 2
[OK] decode 10-bit yuv444p10le bt.2020 gray_diff = 1, color_diff = 2
[OK] encode 12-bit yuv420p12le bt.709  gray_diff = 1, color_diff = 2
[OK] decode 12-bit yuv420p12le bt.709  gray_diff = 1, color_diff = 2
[OK] encode 12-bit yuv420p12le bt.2020 gray_diff = 1, color_diff = 2
[OK] decode 12-bit yuv420p12le bt.2020 gray_diff = 1, color_diff = 2
[OK] encode 12-bit yuv422p12le bt.709  gray_diff = 1, color_diff = 2
[OK] decode 12-bit yuv422p12le bt.709  gray_diff = 1, color_diff = 2
[OK] encode 12-bit yuv422p12le bt.2020 gray_diff = 1, color_diff = 2
[OK] decode 12-bit yuv422p12le bt.2020 gray_diff = 1, color_diff = 2
[OK] encode 12-bit yuv444p12le bt.709  gray_diff = 1, color_diff = 2
[OK] decode 12-bit yuv444p12le bt.709  gray_diff = 1, color_diff = 2
[OK] encode 12-bit yuv444p12le bt.2020 gray_diff = 1, color_diff = 2
[OK] decode 12-bit yuv444p12le bt.2020 gray_diff = 1, color_diff = 2
```

## baseline からの改善

- 10-bit encode: 全条件 `gray_diff <= 1`, `color_diff <= 2` に改善。
- 10-bit decode: 全条件 `gray_diff <= 1`, `color_diff <= 2` に改善。
- 12-bit encode: 全条件 `gray_diff <= 1`, `color_diff <= 2` に改善。
- 12-bit decode: 全条件 `gray_diff <= 1`, `color_diff <= 2` に改善。

## 未解決条件

- 8-bit yuv420p encode/decode color: `color_diff = 3`
- 8-bit yuv422p decode color: `color_diff = 3`
原因候補:

- 8-bit subsampled color は RGB/YUV 係数丸めと chroma 再構成の 8-bit 固有丸めが残っている。
- x86 ASM を修正しない制約により、10-bit/12-bit の該当 SIMD 経路は C fallback にした。

次の調査案:

- 8-bit subsampled color は yuv420/yuv422 の chroma sample 位置、水平/垂直 filter 丸め、RGB24 packed output の丸めを個別に分解する。
