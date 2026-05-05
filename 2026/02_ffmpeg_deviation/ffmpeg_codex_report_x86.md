# FFmpeg x86 RGB to YUV 丸め修正レポート

## 概要

`GBRP10` / `GBRP12` の RGB full range から limited YUV への変換について、x86 assembly 側にも丸め精度修正を入れました。あわせて、commit `a1a821225e` で無効化されていた `GBRP10` / `GBRP12` の x86 override を再有効化しました。

今回の x86 側の変換式は、`libswscale/input.c` に追加済みの C 実装ヘルパーと同じ考え方です。

```c
offset +/- round(abs(sum) * 255 / ((1 << (RGB2YUV_SHIFT - 6)) * ((1 << bpc) - 1)))
```

この経路では `RGB2YUV_SHIFT - 6` が 9 なので、x86 側の分母は以下になります。

- 10-bit: `512 * 1023`
- 12-bit: `512 * 4095`

従来の asm 実装は固定 shift によって実質的に `2^N` 基準で正規化していましたが、修正後は C 実装と同じく入力最大値を `2^N - 1` として扱います。

## 修正ファイル

- `ffmpeg_8.1_src/libswscale/x86/input.asm`
- `ffmpeg_8.1_src/libswscale/x86/swscale.c`

## x86 assembly 側の修正内容

`libswscale/x86/input.asm` に以下を追加・修正しました。

- `LOAD_PLANAR_RGB_WORD`
  - 16-bit planar RGB を scalar load する補助マクロです。
  - big endian 入力では 16-bit word の byte swap を行います。
- `PLANAR_RGB16_TO_YUV14`
  - C 実装の `planar_rgb16_to_yuv14()` と同等の符号付き丸め除算を実装しました。
  - `sum < 0` の場合: `offset - round(abs(sum) * 255 / den)`
  - `sum >= 0` の場合: `offset + round(sum * 255 / den)`
- `planar_rgb_to_y_fn`
  - non-float の 10-bit / 12-bit planar RGB だけ、修正式を使う専用経路にしました。
- `planar_rgb_to_uv_fn`
  - non-float の 10-bit / 12-bit planar RGB だけ、修正式を使う専用経路にしました。
- `div` 命令対策
  - `div` は `rax` / `rdx` を破壊します。
  - x86inc のレジスタ割り当て次第でソースプレーンポインタが `rax` に置かれるため、RGB ソースポインタを stack に退避し、各 loop で復元するようにしました。

既存の vectorized path は 10/12-bit 以外では引き続き使われます。10/12-bit 修正経路は scalar assembly です。理由は、`2^N - 1` 分母での厳密な丸め除算が、従来の固定右 shift だけでは表現できないためです。

## x86 override の再有効化

`libswscale/x86/swscale.c` で、`GBRP10` / `GBRP12` の x86 override case を復元しました。

- AVX2: `INPUT_PLANER_RGB_YUVA_ALL_CASES`
- SSE2: chroma / alpha case
- SSE4: luma / chroma case

これにより、`GBRP10` / `GBRP12` は修正済み C 実装へのフォールバックではなく、x86 側の実装を使う状態に戻っています。

## 検証結果

指定された Docker image で build と encode を実行しました。

```bash
docker run -i -P --name ffmpeg_investigation \
  -v /mnt/c/Users/toruv/OneDrive/work/sample_code:/work/src \
  --rm takuver4/ffmpeg_investigation:rev02 \
  bash -lc 'cd /work/src/2026/02_ffmpeg_deviation && ./scripts/build_ffmpeg.sh && ./scripts/encode.sh'
```

結果:

- `./scripts/build_ffmpeg.sh`: 成功
- `./scripts/encode.sh`: 成功

追加で、10-bit 入力の null encode を CPU flag 別に確認しました。

```text
10-bit null encode passed with cpuflags: 0, sse2, sse4.1, avx2
```

`check_10bit_diff.py` の最終出力は以下です。

```text
[Debug] 8-bit, bt.709: gray_diff = 1, color_diff = 3
[Debug] 8-bit, bt.2020: gray_diff = 1, color_diff = 3
[Debug] 10-bit, bt.709: gray_diff = 1, color_diff = 2
[Debug] 10-bit, bt.2020: gray_diff = 1, color_diff = 2
[Debug] 12-bit, bt.709: gray_diff = 1, color_diff = 2
[Debug] 12-bit, bt.2020: gray_diff = 1, color_diff = 2
Gray maximum difference is 1
OK
Color maximum difference is 3
NG
```

gray 差分は要求条件を満たしています。

- 8-bit gray diff <= 1: pass
- 10-bit gray diff <= 1: pass
- 12-bit gray diff <= 1: pass

一方で、`check_10bit_diff.py` の process exit code は 1 です。理由は、同スクリプトが color 差分も `<= 2` として判定しており、8-bit color path で `color_diff = 3` が出ているためです。

今回再有効化した 10-bit / 12-bit x86 path については、color 差分も `2` に収まっています。

## 未解決事項・補足

今回の修正では、10/12-bit planar RGB について C 実装と同じ数式に合わせることを優先しました。そのため、該当経路は scalar assembly で実装しています。

将来的に性能を重視する場合は、64-bit scalar division を SIMD の reciprocal multiply などに置き換える余地があります。ただし、その場合も以下の性質は維持する必要があります。

- 入力最大値を `2^N - 1` として正規化すること
- `sum` の符号に応じた `offset +/- round(...)` の挙動を保つこと
- C 実装と同等の gray 精度を維持すること
