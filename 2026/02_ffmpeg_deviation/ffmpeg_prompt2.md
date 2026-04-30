# FFmpeg ソースコード解析依頼プロンプト

## 背景

別プロジェクトで FFmpeg を使用した際に、FFmpeg でエンコードした HEVC や AV1 のファイルを DaVinci Resolve や Google Chrome などの別アプリでデコードすると、3/1023 ～ 4/1023 CV 程度の誤差が生じることを経験した。このズレの原因を本格的に調査することにした。

## これまでの流れ

[ffmpeg_prompt.md](./ffmpeg_prompt.md) を参照。
なお、このファイルは参考情報とすること。具体的な指示は以下で行う。

## 具体的な指示

次のステップとしてソースコードの修正により、この問題を解決できるか確認したい。
私は生成AI が効率よく確認できるように、ffmpeg のビルド環境および、誤差評価のスクリプト群を準備した。

以下の手順に従って調査を行って欲しい。

### 調査の概要

* 8-bit/10-bit/12-bit の各bit深度で rgb444 to yuv420 への変換時の誤差を量子化誤差レベルの ±1 に抑えたい
  * なお、±1 は gray (無彩色) の場合であり、color (有彩色) の場合は YCbCr 変換を考慮して誤差は ±2 とする
* 確認用のテストパターンは事前に作成済みである
  * create_src_test_pattern2.py の create_test_pattern_all() で作成
  * 生成物は以下
    * ./img/src_img_v2_08-bit.dpx
    * ./img/src_img_v2_10-bit.dpx
    * ./img/src_img_v2_12-bit.dpx
* この dpx を ffmpeg で yuv420p に変換し、yuv420p に含まれる誤差が小さくなるよう頑張る
* ffmpeg で作成した yuv420p の解析は筆者の作成したスクリプト (./scripts/check_10bit_diff.py) で行う

### Step 1: FFmpeg のソースコード修正

以下に ffmpeg のソースコードを置いた。この内容を解析＆修正して誤差を最小化する。
  * /mnt/c/Users/toruv/OneDrive/work/sample_code/2026/02_ffmpeg_deviation/ffmpeg_8.1_src

### Step 2: FFmpeg のビルドと確認用の yuv420p ファイルの作成

* まず `docker run -it -P --name ffmpeg_investigation -v /mnt/c/Users/toruv/OneDrive/work/sample_code:/work/src --rm takuver4/ffmpeg_investigation:rev02 bash` をコールして docker コンテナ起動
* docker コンテナ内で `cd /work/src/2026/02_ffmpeg_deviation/ && ./scripts/build_ffmpeg.sh` を実行して ffmpeg をビルド
* docker コンテナ内で `cd /work/src/2026/02_ffmpeg_deviation/ && ./scripts/encode.sh` を実行してカスタムした ffmpeg で yuv420p のファイルを生成

### Step 3: FFmpeg で生成した yuv420p ファイルに対して理想値との誤差を確認

* Step 2 の docker コンテナ内で `exit` をしてコンテナを抜ける
* `source /mnt/c/Users/toruv/OneDrive/work/sample_code/.venv_wsl/bin/activate && cd /mnt/c/Users/toruv/OneDrive/work/sample_code/2026/02_ffmpeg_deviation && python3 ./scripts/check_10bit_diff.py` を実行して "OK" となれば成功。"NG" ならば失敗。
* なお、./scripts/check_10bit_diff.py では `check_gray_diff` と `check_color_diff` の2つの評価関数を用意しているが、まずは grey だけ確認する。color は第二弾の調査で確認する。

### Step 4: 上記の Step 1～3 の繰り返し

* 誤差が許容範囲内となるようにソースコードの修正を行う
* Step 4 内で Step 1～3 を繰り返す回数に特に制限は設けないが、流石に 15回を超えても良い結果が出ない場合は断念すること

### Step 5: 調査レポートの作成

* 誤差が許容範囲内に収まる、収まらないに関わらず調査報告結果を Markdown 形式で書くこと
  * もちろん日本語で書く。ただしテクニカルタームは英単語のままで良い
* ファイル名は /mnt/c/Users/toruv/OneDrive/work/sample_code/2026/02_ffmpeg_deviation/ffmpeg_codex_report.md とする
  * 既にファイルが存在していた場合は適当なサフィックスをつけて、ファイルを上書きしないようにすること
