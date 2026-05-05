# FFmpeg rgb444 to yuv420 gray 誤差調査レポート

## 概要

`ffmpeg_8.1_src` を修正し、`scripts/check_10bit_diff.py` の gray 評価が全条件で `OK` になることを確認した。

対象条件は以下。

| bit depth | color matrix | gray diff |
| --- | --- | ---: |
| 8-bit | bt.709 | 1 |
| 8-bit | bt.2020 | 1 |
| 10-bit | bt.709 | 1 |
| 10-bit | bt.2020 | 1 |
| 12-bit | bt.709 | 1 |
| 12-bit | bt.2020 | 1 |

最終結果:

```text
Gray maximum difference is 1
OK
```

なお、`check_10bit_diff.py` では color 評価は無効化されている。参考値として、最終実行時の color diff は 8-bit が 5、10-bit が 14、12-bit が 54/55 だった。

## 調査内容

入力 DPX は FFmpeg で以下の pixel format として decode された。

| 入力 | decode pixel format |
| --- | --- |
| `src_img_v2_08-bit.dpx` | `rgb24` |
| `src_img_v2_10-bit.dpx` | `gbrp10le` |
| `src_img_v2_12-bit.dpx` | `gbrp12le` |

10-bit/12-bit の変換経路は `libswscale/input.c` の planar RGB 入力処理、および x86 SIMD override 経路が該当した。

問題の中心は、RGB full range の N-bit 値を limited YUV へ変換する際に、理想式では入力最大値を `2^N - 1` として正規化すべきところ、既存実装は shift ベースの近似で処理していた点にある。8-bit では `255 == 2^8 - 1` のため問題が目立たないが、10-bit/12-bit では白側で数 CV の過大変換が出る。

初回修正では C 実装を直したが、x86 SIMD 実装が `gbrp10le`/`gbrp12le` の RGB to YUV 関数を上書きしていたため、gray は 10-bit で 3、12-bit で 16 のままだった。

## 修正内容

### `libswscale/input.c`

`planar_rgb16_s*_to_y` と `planar_rgb16_s*_to_uv` の計算で、入力値を `2^bpc - 1` 基準で正規化する helper を追加した。

変更後は、係数適用後の値に対して `255 / ((2^bpc - 1) * 2^(RGB2YUV_SHIFT - 6))` 相当のスケーリングを行い、limited range offset を加える。

これにより、10-bit/12-bit の full-range ramp が white endpoint まで正しく limited YUV の範囲へ収まる。

### `libswscale/x86/swscale.c`

`GBRP10` / `GBRP12` の planar RGB to YUV x86 SIMD override を外し、修正済みの C 実装が使われるようにした。

対象から外した override:

- SSE2 の `GBRP10` / `GBRP12` UV override
- SSE4 の `GBRP10` / `GBRP12` YUV override
- AVX2 の `GBRP10` / `GBRP12` YUVA override

今回の目的は gray 誤差を収束させる検証であり、x86 assembly 側へ同じ数式を移植するところまでは実施していない。

## 検証手順

Docker 内で以下を実行した。

```bash
cd /work/src/2026/02_ffmpeg_deviation/
./scripts/build_ffmpeg.sh
./scripts/encode.sh
```

WSL 側で以下を実行した。

```bash
source /mnt/c/Users/toruv/OneDrive/work/sample_code/.venv_wsl/bin/activate
cd /mnt/c/Users/toruv/OneDrive/work/sample_code/2026/02_ffmpeg_deviation
python3 ./scripts/check_10bit_diff.py
```

最終ログ:

```text
[Debug] 8-bit, bt.709: gray_diff = 1, color_diff = 5
[Debug] 8-bit, bt.2020: gray_diff = 1, color_diff = 5
[Debug] 10-bit, bt.709: gray_diff = 1, color_diff = 14
[Debug] 10-bit, bt.2020: gray_diff = 1, color_diff = 14
[Debug] 12-bit, bt.709: gray_diff = 1, color_diff = 54
[Debug] 12-bit, bt.2020: gray_diff = 1, color_diff = 55
Gray maximum difference is 1
OK
```

## 残課題

今回の修正は gray ramp の許容誤差達成を目的にしたもの。color ramp は評価対象外だったため、chroma subsampling、RGB to YCbCr 係数、色差 plane の丸め、x86 SIMD 実装の同等修正は別途確認が必要。

性能面では `GBRP10` / `GBRP12` の x86 SIMD override を外しているため、該当変換は C 実装になる。実運用または upstream 品質を目指す場合は、`libswscale/x86/input.asm` の `planar_rgb_to_y_fn` / `planar_rgb_to_uv_fn` に同じ `2^bpc - 1` 正規化を移植するのが望ましい。
