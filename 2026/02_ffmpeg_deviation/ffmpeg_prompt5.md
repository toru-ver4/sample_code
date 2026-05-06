# FFmpeg precise 10-bit 誤差修正依頼プロンプト

## 背景

FFmpeg の RGB full range と YUV limited range の相互変換で、10-bit/12-bit を中心に数 code value 程度の誤差が出ている。

過去に `./ffmpeg_8.1_src` を使って `RGB -> YUV` 方向の調査・修正を行ったが、今回はその修正を cherry-pick しない。対象は新しく checkout 済みの fork repo とし、現状の fork から改めて調査・修正・検証する。

## 対象 repo

```text
/mnt/c/Users/toruv/OneDrive/work/sample_code/2026/02_ffmpeg_deviation/ffmpeg_precise_10bit
```

注意:

* `./ffmpeg_8.1_src` は今回の修正対象ではない。
* 過去の `ffmpeg_8.1_src` の commit を前提にしない。
* 既に conflict が生じているため、過去 commit の cherry-pick は行わない。

## 今回の目的

以下の両方向の変換について、誤差を許容範囲内に収めたい。

* RGB full range -> YUV limited range
* YUV limited range -> RGB full range

対象条件は以下。

* bit depth
  * 8-bit
  * 10-bit
  * 12-bit
* color matrix
  * bt.709
  * bt.2020
* YUV pixel format
  * yuv420p
  * yuv422p
  * yuv444p
* ramp
  * gray ramp
  * color ramp

評価は 18 条件 x encode/decode 方向 = 36 評価で行う。

許容誤差:

* gray diff <= 1
* color diff <= 2

なお、今回のテストパターンでは color patch size を事前に調整済みであり、yuv420p/yuv422p の chroma subsampling による評価上の問題は生じない前提である。したがって yuv420p/yuv422p/yuv444p の color ramp はすべて正式な評価対象とする。

## 重要な方針

* x86 ASM は修正しない。
* 最新 FFmpeg ブランチを fork に merge し続けるメンテナンス性を優先する。
* 必要であれば、x86 SIMD override を限定的に無効化し、修正済み C 実装へフォールバックさせてよい。
* ただし、無効化した場合は、どの override をなぜ無効化したのかをレポートに明記すること。
* 評価スクリプトの tolerance を緩めて成功扱いにしてはいけない。
* 評価条件を減らしてはいけない。

## テストパターン

テストパターンは作成済みである。

RGB 入力:

```text
./img/src_img_v2_08-bit.dpx
./img/src_img_v2_10-bit.dpx
./img/src_img_v2_12-bit.dpx
```

YUV reference 入力:

```text
./raw/ref_3840x2160_yuv420p8le_bt.709.yuv
./raw/ref_3840x2160_yuv420p8le_bt.2020.yuv
./raw/ref_3840x2160_yuv422p8le_bt.709.yuv
./raw/ref_3840x2160_yuv422p8le_bt.2020.yuv
./raw/ref_3840x2160_yuv444p8le_bt.709.yuv
./raw/ref_3840x2160_yuv444p8le_bt.2020.yuv
./raw/ref_3840x2160_yuv420p10le_bt.709.yuv
./raw/ref_3840x2160_yuv420p10le_bt.2020.yuv
./raw/ref_3840x2160_yuv422p10le_bt.709.yuv
./raw/ref_3840x2160_yuv422p10le_bt.2020.yuv
./raw/ref_3840x2160_yuv444p10le_bt.709.yuv
./raw/ref_3840x2160_yuv444p10le_bt.2020.yuv
./raw/ref_3840x2160_yuv420p12le_bt.709.yuv
./raw/ref_3840x2160_yuv420p12le_bt.2020.yuv
./raw/ref_3840x2160_yuv422p12le_bt.709.yuv
./raw/ref_3840x2160_yuv422p12le_bt.2020.yuv
./raw/ref_3840x2160_yuv444p12le_bt.709.yuv
./raw/ref_3840x2160_yuv444p12le_bt.2020.yuv
```

## 検証手順

Docker コンテナを起動:

```bash
docker run -it -P --name ffmpeg_investigation -v /mnt/c/Users/toruv/OneDrive/work/sample_code:/work/src --rm takuver4/ffmpeg_investigation:rev02 bash
```

Docker 内で FFmpeg をビルド:

```bash
cd /work/src/2026/02_ffmpeg_deviation/
./scripts/build_ffmpeg.sh
```

Docker 内で RGB -> YUV 評価用データを生成:

```bash
cd /work/src/2026/02_ffmpeg_deviation/
./scripts/encode.sh
```

Docker 内で YUV -> RGB 評価用データを生成:

```bash
cd /work/src/2026/02_ffmpeg_deviation/
./scripts/decode.sh
```

Docker から抜けたあと、WSL 側で統合評価を実行:

```bash
source /mnt/c/Users/toruv/OneDrive/work/sample_code/.venv_wsl/bin/activate
cd /mnt/c/Users/toruv/OneDrive/work/sample_code/2026/02_ffmpeg_deviation
python3 ./scripts/check_10bit_diff.py
```

Python 実行時は必ず既存の `.venv_wsl` を使うこと。追加の仮想環境を作らないこと。

## 修正前 baseline

素の `ffmpeg_precise_10bit` での baseline は以下。

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
[NG] encode 10-bit yuv420p10le bt.709  gray_diff = 3, color_diff = 4
[NG] decode 10-bit yuv420p10le bt.709  gray_diff = 3, color_diff = 4
[NG] encode 10-bit yuv420p10le bt.2020 gray_diff = 3, color_diff = 5
[NG] decode 10-bit yuv420p10le bt.2020 gray_diff = 3, color_diff = 4
[NG] encode 10-bit yuv422p10le bt.709  gray_diff = 3, color_diff = 4
[NG] decode 10-bit yuv422p10le bt.709  gray_diff = 3, color_diff = 4
[NG] encode 10-bit yuv422p10le bt.2020 gray_diff = 3, color_diff = 5
[NG] decode 10-bit yuv422p10le bt.2020 gray_diff = 3, color_diff = 4
[NG] encode 10-bit yuv444p10le bt.709  gray_diff = 3, color_diff = 4
[NG] decode 10-bit yuv444p10le bt.709  gray_diff = 3, color_diff = 4
[NG] encode 10-bit yuv444p10le bt.2020 gray_diff = 3, color_diff = 5
[NG] decode 10-bit yuv444p10le bt.2020 gray_diff = 3, color_diff = 4
[NG] encode 12-bit yuv420p12le bt.709  gray_diff = 16, color_diff = 17
[NG] decode 12-bit yuv420p12le bt.709  gray_diff = 16, color_diff = 18
[NG] encode 12-bit yuv420p12le bt.2020 gray_diff = 16, color_diff = 17
[NG] decode 12-bit yuv420p12le bt.2020 gray_diff = 16, color_diff = 18
[NG] encode 12-bit yuv422p12le bt.709  gray_diff = 16, color_diff = 17
[NG] decode 12-bit yuv422p12le bt.709  gray_diff = 16, color_diff = 18
[NG] encode 12-bit yuv422p12le bt.2020 gray_diff = 16, color_diff = 17
[NG] decode 12-bit yuv422p12le bt.2020 gray_diff = 16, color_diff = 18
[NG] encode 12-bit yuv444p12le bt.709  gray_diff = 16, color_diff = 17
[NG] decode 12-bit yuv444p12le bt.709  gray_diff = 16, color_diff = 18
[NG] encode 12-bit yuv444p12le bt.2020 gray_diff = 16, color_diff = 17
[NG] decode 12-bit yuv444p12le bt.2020 gray_diff = 16, color_diff = 18
```

## 調査・修正対象の目安

主な対象は `libswscale` を想定する。

RGB -> YUV 方向:

* `libswscale/input.c`
* `libswscale/swscale.c`
* `libswscale/utils.c`
* 必要であれば `libswscale/x86/swscale.c` の override 選択

YUV -> RGB 方向:

* `libswscale/output.c`
* `libswscale/swscale.c`
* `libswscale/utils.c`
* 必要であれば `libswscale/x86/swscale.c` の override 選択

特に、N-bit の full range RGB または limited range YUV を扱う際に、入力最大値を `2^N - 1` として正規化すべきところが shift ベース近似になっていないか、丸め位置や係数スケーリングが 8-bit 前提になっていないかを確認すること。

## 成功条件

`python3 ./scripts/check_10bit_diff.py` の 36 評価で、すべて以下を満たすこと。

* gray_diff <= 1
* color_diff <= 2

## 失敗時の扱い

すべてを許容範囲に収められない場合でも、以下を明確にすること。

* どの条件が改善したか
* どの条件が未解決か
* 未解決の原因候補
* 今回の方針、特に x86 ASM を修正しない方針の制約によるものか
* 追加で必要な検証

## レポート作成

調査・修正・検証結果を Markdown で報告すること。

レポート名:

```text
/mnt/c/Users/toruv/OneDrive/work/sample_code/2026/02_ffmpeg_deviation/ffmpeg_codex_report_prompt5.md
```

既に同名ファイルが存在する場合は、サフィックスを付けて上書きしないこと。

レポートに必ず含める内容:

* 修正したファイル一覧
* encode 方向と decode 方向で、それぞれどの関数・処理を修正したか
* 数式上の変更点
* x86 SIMD override を無効化した場合、その対象と理由
* 修正前 baseline と修正後結果の比較
* 最終的な `check_10bit_diff.py` の出力
* 未解決条件が残る場合、その原因候補と次の調査案
