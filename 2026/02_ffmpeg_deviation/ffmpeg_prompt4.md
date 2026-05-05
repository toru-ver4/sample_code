# FFmpeg x86 assembly 側修正依頼プロンプト

## 背景

`ffmpeg_prompt2.md` の指示に従って FFmpeg の rgb444 to yuv420 変換誤差を調査・修正した結果、`ffmpeg_codex_report.md` の結果を得た。

前回の修正は commit 済みである。

* 対象 repo: `/mnt/c/Users/toruv/OneDrive/work/sample_code/2026/02_ffmpeg_deviation/ffmpeg_8.1_src`
* 前回 commit: `a1a821225e swscale: improve RGB to YUV rounding accuracy`

前回の結果では、`libswscale/input.c` の C 実装は修正され、gray 誤差は許容範囲内になった。

しかし、`libswscale/x86/swscale.c` で `GBRP10` / `GBRP12` の x86 SIMD override を無効化し、修正済み C 実装にフォールバックさせている。これは性能面・upstream 品質の観点で不十分である。

## 今回の目的

x86 assembly / SIMD 側にも前回 C 実装と同等の修正を入れ、`GBRP10` / `GBRP12` の x86 SIMD override を再有効化した状態で、gray 誤差を許容範囲内に収めたい。

## 具体的な指示

1. まず前回 commit の内容を確認すること。

   ```bash
   cd /mnt/c/Users/toruv/OneDrive/work/sample_code/2026/02_ffmpeg_deviation/ffmpeg_8.1_src
   git show a1a821225e
   ```

2. `libswscale/input.c` に入れた C 実装の数式・丸め・正規化方法を理解すること。

   特に、RGB full range の N-bit 値を limited YUV に変換する際、入力最大値を `2^N - 1` として正規化する点を x86 SIMD 側にも反映すること。

3. `libswscale/x86/swscale.c` で無効化された `GBRP10` / `GBRP12` の x86 SIMD override を、可能な限り元に戻すこと。

   ただし、単に override を戻すだけではなく、対応する x86 assembly 実装を修正し、C 実装と同等の精度になるようにすること。

4. 主な調査・修正対象は以下を想定する。

   * `libswscale/x86/swscale.c`
   * `libswscale/x86/input.asm`
   * 必要であれば `libswscale/x86/` 配下の関連ファイル

   特に `planar_rgb_to_y_fn` / `planar_rgb_to_uv_fn` および `GBRP10` / `GBRP12` 向けの RGB to YUV 変換経路を重点的に確認すること。

5. 今回は、`GBRP10` / `GBRP12` の x86 SIMD override を外して C 実装へフォールバックさせる対応は禁止する。

   やむを得ず一部の SIMD 経路を無効化する場合は、なぜ assembly 修正が不可能または不合理だったのか、具体的な理由をレポートに書くこと。

6. 修正後、以下の手順でビルド・検証すること。

   Docker コンテナを起動:

   ```bash
   docker run -it -P --name ffmpeg_investigation -v /mnt/c/Users/toruv/OneDrive/work/sample_code:/work/src --rm takuver4/ffmpeg_investigation:rev02 bash
   ```

   Docker 内で実行:

   ```bash
   cd /work/src/2026/02_ffmpeg_deviation/
   ./scripts/build_ffmpeg.sh
   ./scripts/encode.sh
   ```

   Docker から抜けた後、WSL 側で実行:

   ```bash
   source /mnt/c/Users/toruv/OneDrive/work/sample_code/.venv_wsl/bin/activate
   cd /mnt/c/Users/toruv/OneDrive/work/sample_code/2026/02_ffmpeg_deviation
   python3 ./scripts/check_10bit_diff.py
   ```

7. `check_10bit_diff.py` が `OK` になることを目標とする。

   少なくとも以下を満たすこと。

   * 8-bit gray diff <= 1
   * 10-bit gray diff <= 1
   * 12-bit gray diff <= 1
   * `GBRP10` / `GBRP12` の x86 SIMD override が無効化されたままになっていないこと

8. 可能であれば、C 実装と x86 SIMD 実装の出力差分がない、または許容範囲内であることを追加で確認すること。

9. 調査・修正・検証結果を Markdown で報告すること。

   レポート名は以下とする。

   ```text
   /mnt/c/Users/toruv/OneDrive/work/sample_code/2026/02_ffmpeg_deviation/ffmpeg_codex_report_x86.md
   ```

   既に存在する場合はサフィックスを付けて上書きしないこと。

## レポートに必ず含める内容

* 修正したファイル一覧
* x86 assembly 側でどの関数・マクロを修正したか
* C 実装との数式上の対応関係
* `GBRP10` / `GBRP12` の x86 SIMD override を再有効化できたか
* `check_10bit_diff.py` の最終結果
* 失敗した場合は、どこまで分かっていて、何が未解決か
