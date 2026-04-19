# FFmpeg ソースコード解析依頼プロンプト

以下は、FFmpeg のソースコード解析を別の生成 AI に依頼するためのプロンプトです。  
そのまま渡して使えるように書いてあります。

---

あなたは FFmpeg の C ソースコードを読むことに強い解析 AI です。  
私が観測した現象について、FFmpeg のソースコードを根拠に原因候補を絞り込み、必要なら修正方針まで提案してください。

## 目的

FFmpeg で静止画から 10-bit HEVC/AV1 を作ると、デコード後の Code Value がわずかにズレます。  
このズレが FFmpeg ソースコード上のどこで生じうるのかを、できるだけ具体的に特定してください。

特に知りたいのは次の 2 点です。

1. `full range` の静止画入力を `limited range` の YUV 動画へ変換する経路で、FFmpeg がどこでどんな数式・丸め・係数を使っているか
2. その処理に問題があるなら、コマンドライン引数で回避できるのか、ソース修正が必要なのか

## 前提となる観測結果

### 観測 1: 静止画 -> FFmpeg エンコード -> デコードで微小なズレが見える

- 対象は主に `PNG -> HEVC/AV1` のケース
- ズレ量は小さいが、10-bit ramp をグラフ化すると目視で分かる
- `gamma 2.2 と 2.4 を取り違えた` レベルの大きな差ではない
- 体感としては `Full <-> Limited` 変換時の係数や丸めの微妙な差に近い

### 観測 2: Encode/Decode を同じソフトで完結させると大きな問題は出にくい

- Resolve で Encode -> Resolve で Decode: 許容範囲
- FFmpeg で Encode -> FFmpeg で Decode: 許容範囲
- 片方を FFmpeg、もう片方を Resolve にすると、最大でおよそ `4/1023` 程度の差が見えた
- この段階では FFmpeg と Resolve のどちらが間違っているか断定できなかった

### 観測 3: 生の I010 YUV から FFmpeg でエンコードすると、問題のズレは再現しない

- 自前で `yuv420p10le / I010` の生データを作成
- その I010 を FFmpeg で HEVC エンコード
- 生成 bitstream を `de265` でデコードし、自前で I010 として確認
- このケースでは、これまで静止画起点で見えていたズレを確認できなかった

### 暫定結論

怪しいのは `FFmpeg のエンコーダそのもの` というより、

- `full range` の静止画データを読み込む経路
- そこから `limited range` の `yuv420p10le` へ落とす経路
- その際の色域変換、レンジ変換、係数、丸め、ディザ、色レンジ metadata の扱い

です。

## 実験条件

### 使用していた FFmpeg コマンドの要点

静止画入力からのエンコードでは概ね次のような条件です。

```bash
ffmpeg \
  -loop 1 \
  -color_primaries bt709 \
  -color_trc bt709 \
  -colorspace bt709 \
  -framerate 24 \
  -t 5 \
  -i src_img.png \
  -c:v libx265 \
  -pix_fmt yuv420p10le \
  -color_primaries bt709 \
  -color_trc bt709 \
  -colorspace bt709 \
  -qp 0 \
  out.mp4
```

AV1 では `-c:v libsvtav1` を使用します。

### デコード時の条件

FFmpeg で静止画へ戻すときは次のようなフィルタを使っています。

```bash
ffmpeg \
  -i out.mp4 \
  -map 0:v:0 \
  -frames:v 1 \
  -vf scale=in_range=limited:out_range=full \
  -pix_fmt rgb48be \
  out.png
```

### 比較した入力

- `src_img.png`
- `src_img.tif`
- `src_img.dpx`
- `src_img.exr`

ただし、最も怪しい本命は `PNG` を含む「静止画入力 -> FFmpeg 内で YUV420P10 limited 化」の経路です。

## 解析で特に見てほしい点

以下を優先して追ってください。

### 1. 静止画デコーダが AVFrame に入れる range / colorspace 情報

候補:

- `libavcodec/pngdec.c`
- `libavcodec/tiff.c`
- `libavcodec/exr.c`
- `libavcodec/dpx.c`
- 必要に応じて `AVFrame.color_range` や関連フィールドを設定している箇所全般

見てほしいこと:

- PNG/TIFF/DPX/EXR の各デコーダが、出力 `AVFrame` に対して `color_range` をどう設定しているか
- PNG の場合、RGB 入力は `AVCOL_RANGE_JPEG` 相当として扱われているか
- range 未設定のまま後段に流れていないか
- `gAMA`, `cICP`, `sRGB`, ICC profile などの扱いが今回のケースに影響しうるか

### 2. RGB 静止画から yuv420p10le へ変換する実体

候補:

- `libswscale/*`
- 特に `libswscale/swscale.c`
- `libswscale/utils.c`
- `libswscale/output.c`
- `libswscale/input.c`
- range conversion, matrix, rounding, dithering 周辺

見てほしいこと:

- RGB full range -> YUV limited range のとき、どの関数で係数が決まるか
- BT.709 指定時の行列とレンジ変換係数がどこで決まるか
- 10-bit 量子化時の丸め規則
- `sws_getContext` / `sws_setColorspaceDetails` / `SwsContext` のどのパラメータが今回の経路を支配するか
- RGB 入力の `full` 扱いと、YUV 出力の `limited` 扱いがコード上でどう接続されるか
- 1 LSB 前後のズレを生む実装上の可能性があるか

### 3. ffmpeg CLI 側がレンジ情報をどう解釈しているか

候補:

- `fftools/ffmpeg*.c`
- `fftools/ffmpeg_filter.c`
- `fftools/ffmpeg_mux.c`
- `fftools/ffmpeg_demux.c`
- `fftools/cmdutils.c`

見てほしいこと:

- `-color_primaries`, `-color_trc`, `-colorspace` は設定しているが、`-color_range` を指定していない状態で何が起こるか
- image demux/decode -> filtergraph -> encoder の間で `color_range` が暗黙にどう扱われるか
- CLI 引数だけで `full -> limited` の挙動を明示制御できるか
- `-vf scale=in_range=full:out_range=limited` のように明示した場合、今回の曖昧さを回避できる可能性があるか

### 4. libx265 / libsvtav1 ラッパーに起因する可能性の切り分け

候補:

- `libavcodec/libx265.c`
- `libavcodec/libsvtav1.c`

見てほしいこと:

- encoder wrapper が受け取った `AVFrame` をそのまま渡しているだけか
- ここで range 変換や追加の量子化が入る余地があるか
- ただし、生 I010 入力ではズレが出ないので、主因はここではないと予想している

## あなたに期待する出力

次の構成で答えてください。

### A. 最有力の原因候補

- 1 位から順位付きで 3 件以内
- それぞれについて、どのソースファイルのどの関数が怪しいかを書く
- 「なぜそこが怪しいのか」を、私の観測結果と結びつけて説明する

### B. データフローの整理

次の流れを、関数名や構造体名ベースで説明してください。

`PNG/TIFF/DPX/EXR decode`
-> `AVFrame`
-> `libswscale or filtergraph`
-> `yuv420p10le`
-> `libx265/libsvtav1`

特に `color_range`, `colorspace`, `color_trc`, `primaries` がどこで設定・参照・変換されるかを明記してください。

### C. ズレの数理的な説明

可能なら、RGB full -> YUV limited 10-bit 変換で

- 係数の選び方
- 丸め
- clipping
- chroma subsampling

のどれで `1〜4 code value / 1023` 程度の差が出うるかを説明してください。

### D. 回避策

次の 2 系統に分けて提案してください。

1. ソース修正なしで試せる FFmpeg コマンドライン回避策
2. FFmpeg ソース修正が必要な場合の修正候補

回避策には、できれば具体的なコマンド例も添えてください。

### E. 確認実験案

私が追加で行うべき最小限の検証を 3 件以内で提案してください。  
それぞれについて、

- 何を比較するか
- どの結果なら仮説を支持するか

を明確に書いてください。

## 制約

- 推測だけで断定しないでください
- 可能な限り「ソースコード上の根拠」を示してください
- 不明な点は「未確認」と明記してください
- もし FFmpeg のバージョン差分が重要なら、その可能性も指摘してください
- 私の観測では `FFmpeg 8.0.1` と書いているメモと `FFmpeg 8.1` と書いているメモが混在しています。必要なら「どの版を前提に解析しているか」を最初に明示してください

## 補足

私の手元の観測からは、少なくとも以下の仮説が濃厚です。

- `生 I010 -> FFmpeg encode` ではズレが出ない
- `静止画 full range RGB -> FFmpeg 内部で limited YUV420P10 化` の経路でズレが出る
- したがって、`libx265` 自体より `decode 後の AVFrame 属性` または `swscale / format conversion` の方が怪しい

この仮説が正しいかどうかも含めて、反証可能な形で検討してください。

---

必要なら、最後に「最短で読むべきファイル一覧」を 5〜10 個に絞って提示してください。
