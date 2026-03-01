# 1. 背景

* 筆者はこれまで、HDR の動画・静止画コンテンツを [Windows 上で正しく表示する方法](https://trev16.hatenablog.com/entry/2024/07/30/195204) について調査をしてきた
* [PNG の検証](https://trev16.hatenablog.com/entry/2025/09/27/154448) をしている中で CLLI のメタデータによってコンテンツの見え方が大きく変わることに気づき、HDR コンテンツのメタデータが表示に与える影響を改めて確認したいと考えた
* 確認作業のためには、そもそも 正しくメタデータを付与する 作業が必要となる
* それを行うことにした

# 2. 目的

* HDR の動画・静止画コンテンツに [CICP](https://www.w3.org/TR/png-3/#cICP-chunk)、[MDCV](https://www.w3.org/TR/png-3/#mDCV-chunk)、[CLLI](https://www.w3.org/TR/png-3/#cLLI-chunk) のメタデータを正しく付与する方法をまとめる
* 使用する動画・静止画のフォーマットは以下の通り
  * 動画
    * コーデック: HEVC、AV1
    * コンテナ: MP4、MOV
  * 静止画
    * AVIF、PNG
* [CICP](https://www.w3.org/TR/png-3/#cICP-chunk)、[MDCV](https://www.w3.org/TR/png-3/#mDCV-chunk)、[CLLI](https://www.w3.org/TR/png-3/#cLLI-chunk) のメタデータの詳細については以下の記事を参照

[https://trev16.hatenablog.com/entry/2026/02/14/204125:embed:cite]

# 3. 結論

#### 3.1. まとめ

* HEVC、AV1、AVIF、PNG の 4 フォーマットに対してメタデータの埋め込みに成功した
  * ただし、AVIF だけは MDCV の埋め込みが [libavif 側で未実装](https://github.com/AOMediaCodec/libavif/blob/v1.3.0/src/write.c#L723) だったため実現できなかった
  * メタデータの内容が正しいことは [gpac](https://wiki.gpac.io/Filters/Filters/)/[MP4Box](https://wiki.gpac.io/MP4Box/MP4Box/)/[pngcheck](https://github.com/pnggroup/pngcheck) などをパーサー代わりに使いテストコードを作成して確認した ((ただし、一部の規格文書は金銭的な都合で買えておらず、テスト内容が正しいことを裏付ける公式なデータは無い)) ((本当は買うべきなんだろうけど、Nintendo Switch 2 本体が買えるくらいの値段なので買うのは厳しい))
* メタデータの埋め込みは 2026年2月時点では簡単ではなく、HEVC、AV1、PNG は図1 のように中間ファイルの生成が必要であった
  * 加えてソースコードに若干の修正も必要であった（詳細は「4. 作業環境」の項目を参照）

<figure class="figure-image figure-image-fotolife" title="図1. HEVC、AV1、PNG にメタデータを埋め込む際の処理概要">[f:id:takuver4:20260301130946p:plain:w600]<figcaption>図1. HEVC、AV1、PNG にメタデータを埋め込む際の処理概要</figcaption></figure>

#### 3.2. 筆者が理解したこと

* MP4/MOV の CICP、MDCV、CLLI 情報は bitstream の情報から生成可能
  * 筆者はこれまで MP4/MOV コンテナ生成時に別途 CICP、MDCV、CLLI 情報を与えるものだと勘違いしていた
* [gpac](https://wiki.gpac.io/Filters/Filters/) で bitstream のメタ情報をダンプした場合、[人間が解釈しやすいように変換](https://github.com/gpac/gpac/blob/v26.02.0/src/filters/inspect.c#L501-L529) してくれてる
  * 読みやすい一方で、規格文書と値が異なるのでテストコードを作成する際は注意が必要
* AVIF は bitstream と AVIF コンテナとでメタデータの値が一致しない
  * bitstream の CICP は 2/2/2 に固定化され、MDCV と CLLI の情報は埋め込まれない（libavif の仕様。詳細は後述）
  * もしも AVIF から bitstream だけを抽出して何らかの処理を行うことがあれば注意が必要 ((そんな使い方は誰もしないと思うが))
* 筆者が調べたところ PNG に `mDCV`、`cLLI` chunk を埋め込めるツールは FFmpeg のみであった
  * `mDCV`、`cLLI` chunk を積極的に使いたいと思っている人はほとんどいない？ 
* 関連情報を調べていた所 [ITU-R H.274](https://www.itu.int/rec/T-REC-H.274/en) で追加された Content colour volume は定義が分かりやすくて良かった
  * MDCV、CLLI は今後は Content colour volume に置き換わるのでは、と勝手に予想している

#### 3.3. 作成したファイル

様々な値のメタデータを埋め込んだ、計126個のファイルを以下に添付しておく。

[https://drive.google.com/file/d/15tASRelJ3flvfpi9eR3prR0ZQnU1vCNj/view?usp=drive_link:embed:cite]


# 4. 作業環境

ファイル生成は以下の環境で行った。

<div style="text-align: center; margin: 1.5em 0;">
  <div style="font-weight: bold; margin-bottom: 0.5em;">
    表1. 検証環境および使用ソフトウェアのバージョン
  </div>
  <table style="margin-left: auto; margin-right: auto; border-collapse: collapse;">
    <thead>
      <tr>
        <th>名称</th>
        <th>バージョン</th>
        <th>備考</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>OS</td>
        <td>Debian (Trixie)</td>
        <td>
          <a href="https://hub.docker.com/layers/library/python/3.13-slim-trixie/images/sha256-7f7f96867e2bf7bee2e2d11d5d46d65760f3eca347141a2f1410d98088b0e338">
            python:3.13-slim-trixie
          </a>
          を使用
        </td>
      </tr>
      <tr>
        <td>FFmpeg</td>
        <td>v8.0.1 相当</td>
        <td>
          bitstream の生成および PNG の作成で使用。<br>
          PNG の mDCV chunk の
          <a href="https://github.com/toru-ver4/FFmpeg_png_mdcv/commit/eb78d47428cbae85f6e01d702ca69802764ebdd2">
            アドレス計算ミスの修正
          </a>
          を適用したものをビルドした。
        </td>
      </tr>
      <tr>
        <td>gpac</td>
        <td>v26.02.0 相当</td>
        <td>
          bitstream からのコンテナ生成、および bitstream / コンテナ のメタデータ出力で使用。<br>
          <a href="https://github.com/toru-ver4/gpac/commit/0df998e4ad40833d64d256462c34e82867a33917">
            RGB to GBR の並べ替えミスの修正
          </a>
          を適用したものをビルドした。
        </td>
      </tr>
    </tbody>
  </table>
</div>

# 5. 詳細 (ファイル生成)

#### 5.1. HEVC、AV1、AVIF、PNG の選定理由

HDRに対応したフォーマットは数多くある。今回は動画・静止画フォーマットの中で HEVC、AV1、AVIF、PNG を選んだ条件は以下である。

* ST 2084 対応
* MDCV / CLLI 対応
* Chromium 系ブラウザで表示可能

筆者が各種フォーマットに対して調査した結果を以下の表に示す。

<div style="text-align: center; margin: 1.5em 0;">
  <div style="font-weight: bold; margin-bottom: 0.5em;">
    表2. 各画像・映像フォーマットにおける HDR 関連機能対応状況
  </div>
  <table style="margin-left: auto; margin-right: auto; border-collapse: collapse;">
    <thead>
      <tr>
        <th>フォーマットの種類</th>
        <th>選定</th>
        <th>ST 2084 対応</th>
        <th>MDCV / CLLI 対応</th>
        <th>Chromium 系ブラウザで表示可能</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>HEVC</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>AV1</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>AVIF</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>PNG</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>VVC / H.266</td>
        <td>-</td>
        <td>✓</td>
        <td>✓</td>
        <td>-</td>
      </tr>
      <tr>
        <td>VP9</td>
        <td>-</td>
        <td>△ (※)</td>
        <td>△ (※)</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>Ultra HDR</td>
        <td>-</td>
        <td>✓</td>
        <td>-</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>JPEG XL</td>
        <td>-</td>
        <td>✓</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>HEIF</td>
        <td>-</td>
        <td>✓</td>
        <td>✓</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>
</div>

※bitstream としては非対応<span style="color: #ff5252">&lt;要出典&gt;</span>。[WebM コンテナ](https://www.webmproject.org/docs/container/) のみでサポート

#### 5.2. 用意したメタデータの組み合わせ

CICP、MDCV、CLLI に埋め込むパラメータは以下の15通りを用意し、後述の動作確認で意図した値が書き込まれているか確認できるようにした。ただし、AVIF に関しては [libavif に MDCV を埋め込む実装が存在しなかった](https://github.com/AOMediaCodec/libavif/blob/v1.3.0/src/write.c#L723) ので MDCV は埋め込んでいない。ご了承いただきたい。

<div style="text-align: center; margin: 1.5em 0;">
  <div style="font-weight: bold; margin-bottom: 0.5em;">
    表3. CICP と HDR 静的メタデータ（MDCV / CLLI）の組み合わせ例
  </div>
  <table style="margin-left: auto; margin-right: auto; border-collapse: collapse;">
    <thead>
      <tr>
        <th>No</th>
        <th>CICP ((4番目の Video Full Range Flag は、動画は 0 (Limited) を、静止画は 1 (Full) を設定した))</th>
        <th>MDCV RGBW</th>
        <th>MDCV Luminance ((Mastering display maximum luminance のみを設定、Mastering display minimum luminance は 0 固定とした))</th>
        <th>CLLI Luminance ((MaxCLL と MaxFALL の値は同じ値とした))</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td>9-16-9-*</td>
        <td>ITU-R BT.709</td>
        <td>100</td>
        <td>100</td>
      </tr>
      <tr>
        <td>2</td>
        <td>9-16-9-*</td>
        <td>ITU-R BT.709</td>
        <td>100</td>
        <td>10000</td>
      </tr>
      <tr>
        <td>3</td>
        <td>9-16-9-*</td>
        <td>ITU-R BT.709</td>
        <td>100</td>
        <td>Not present</td>
      </tr>
      <tr>
        <td>4</td>
        <td>9-16-9-*</td>
        <td>ITU-R BT.709</td>
        <td>10000</td>
        <td>100</td>
      </tr>
      <tr>
        <td>5</td>
        <td>9-16-9-*</td>
        <td>ITU-R BT.709</td>
        <td>10000</td>
        <td>10000</td>
      </tr>
      <tr>
        <td>6</td>
        <td>9-16-9-*</td>
        <td>ITU-R BT.709</td>
        <td>10000</td>
        <td>Not present</td>
      </tr>
      <tr>
        <td>7</td>
        <td>9-16-9-*</td>
        <td>ITU-R BT.2020</td>
        <td>100</td>
        <td>100</td>
      </tr>
      <tr>
        <td>8</td>
        <td>9-16-9-*</td>
        <td>ITU-R BT.2020</td>
        <td>100</td>
        <td>10000</td>
      </tr>
      <tr>
        <td>9</td>
        <td>9-16-9-*</td>
        <td>ITU-R BT.2020</td>
        <td>100</td>
        <td>Not present</td>
      </tr>
      <tr>
        <td>10</td>
        <td>9-16-9-*</td>
        <td>ITU-R BT.2020</td>
        <td>10000</td>
        <td>100</td>
      </tr>
      <tr>
        <td>11</td>
        <td>9-16-9-*</td>
        <td>ITU-R BT.2020</td>
        <td>10000</td>
        <td>10000</td>
      </tr>
      <tr>
        <td>12</td>
        <td>9-16-9-*</td>
        <td>ITU-R BT.2020</td>
        <td>10000</td>
        <td>Not present</td>
      </tr>
      <tr>
        <td>13</td>
        <td>9-16-9-*</td>
        <td>Not present</td>
        <td>Not present</td>
        <td>100</td>
      </tr>
      <tr>
        <td>14</td>
        <td>9-16-9-*</td>
        <td>Not present</td>
        <td>Not present</td>
        <td>10000</td>
      </tr>
      <tr>
        <td>15</td>
        <td>9-16-9-*</td>
        <td>Not present</td>
        <td>Not present</td>
        <td>Not present</td>
      </tr>
    </tbody>
  </table>
</div>

#### 5.3. 各種フォーマットにメタデータを埋め込む手順

ここから先は、各種フォーマットにメタデータをどう埋め込んだかを解説する。
AVIF 以外は少々特殊な手順を踏んでいる。その理由も含めて説明する。

#### 5.4. HEVC、AV1

##### 5.4.1. 概要

HEVC、AV1 のファイルは MP4 のコンテナに入れる形とした。実は MOV コンテナも作成して確認をしていたのだが、特に差異を確認できなかったので本記事では MP4 コンテナを使う前提で説明をする。

MP4 ファイルの作成は下図のように [FFmpeg](https://ffmpeg.org/ffmpeg.html) で bitstream を作成してから [MP4Box](https://github.com/gpac/gpac/wiki/MP4Box) を使う方式を取った。

<figure class="figure-image figure-image-fotolife" title="図xx. HEVC、AV1 の MP4 ファイル作成手順">[f:id:takuver4:20260217212250p:plain:w650]<figcaption>図xx. HEVC、AV1 の MP4 ファイル作成手順</figcaption></figure>

メタデータは FFmpeg のコマンドライン引数として与え、MP4Box ではメタデータを与えていない。これは <span style="color: #ff5252">MP4 コンテナの CICP、MDCV、CLLI の Box 情報は bitstream に含まれるデータから生成される</span>ことを意味する。

FFmpeg のみで完結せずに MP4Box を使用した理由は、FFmpeg では MDCV、CLLI の書き込みが上手く行かなかったからである。AI を使いながら調べて分かったことは以下。

* FFmpeg では [libavformat/movenc.c](https://github.com/FFmpeg/FFmpeg/blob/master/libavformat/movenc.c) にて MP4 コンテナにデータを書き込む処理を行っている
* movenc.c には `mov_write_mdcv_tag` や `mov_write_clli_tag` などの関数があり、ソースコード上は書き込めるように見える
* しかし、今回のようにソースを静止画の PNG ファイルにすると [side_data](https://github.com/FFmpeg/FFmpeg/blob/33b215d1554a14e87416a24f8e6034312e629af7/libavformat/movenc.c#L2656-L2658) に情報が入らず書き込みが行われない

ということで代替案として [MP4Box](https://github.com/gpac/gpac/wiki/MP4Box) コマンドを使うことにした。MP4Box は ISOBMFF を処理するためのコマンドラインツールである。

##### 5.4.2. コマンドライン引数

続いて使用した FFmpeg および MP4Box のコマンドライン引数を説明する。FFmpeg で使用したコマンドライン引数は以下の通りである。入出力のファイル名は筆者の環境のままなので若干読みづらいが、そこはご容赦願いたい。

```bash
# HEVC
ffmpeg -hide_banner \
  -loop 1 \
  -color_primaries bt2020 \
  -color_trc smpte2084 \
  -colorspace bt2020nc \
  -framerate 24 \
  -t 10 \
  -i ./src_img/1920x1080_ST2084_Rec.2020.png \
  -c:v libx265 \
  -x265-params "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited:master-display=G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(100000000,0):max-cll=10000,10000" \
  -pix_fmt yuv420p10le \
  -color_primaries bt2020 \
  -color_trc smpte2084 \
  -colorspace bt2020nc \
  -qp 0 \
  -f hevc \
  hdr_media/hevc_mdcv-p-ITU-R\ BT.2020_mdcv-l-10000_clli-10000.h265

# -------------------------------------------------------------------------------

# AV1
ffmpeg -hide_banner \
  -loop 1 \
  -color_primaries bt2020 \
  -color_trc smpte2084 \
  -colorspace bt2020nc \
  -framerate 24 \
  -t 10 \
  -i ./src_img/1920x1080_ST2084_Rec.2020.png \
  -c:v libsvtav1 \
  -svtav1-params "crf=1:mastering-display=G(0.17,0.797)B(0.131,0.046)R(0.708,0.292)WP(0.3127,0.329)L(10000,0):content-light=10000,10000:color-primaries=9:transfer-characteristics=16:matrix-coefficients=9:color-range=0" \
  -color_primaries bt2020 \
  -color_trc smpte2084 \
  -colorspace bt2020nc \
  -pix_fmt yuv420p10le \
  -f obu \
  hdr_media/av1_mdcv-p-ITU-R\ BT.2020_mdcv-l-10000_clli-10000.obu
```

いくつかの引数について、以下に箇条書きで補足説明をしておく。

* `-color_primaries`、`-color_trc`、`-colorspace`は FFmpeg に <span style="color: #ff5252">余計な色変換を行わせない</span> ため指定
* `-x265-params`は CICP、MDCV、CLLI の情報を bitstream に埋め込むために指定。意味は [ x265 のドキュメント](https://x265.readthedocs.io/en/master/cli.html) を参照
  * 補足だが NVENC ではメタデータが埋め込めなかったので x265 を使用した
* `-svtav1-params`は CICP、MDCV、CLLI の情報を bitstream に埋め込むために指定。意味は [SVT-AV1 のドキュメント](https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/master/Docs/Parameters.md) を参照
  * 補足だが libaom ではメタデータが埋め込めなかったので SVT-AV1 を使用した

次に MP4Box で使用したコマンドライン引数を以下に示す。

```bash
# HEVC
MP4Box -new \
  -add hdr_media/hevc_mdcv-p-ITU-R\ BT.2020_mdcv-l-10000_clli-10000.h265:fmt=hevc:fps=24 \
  ./hdr_media/hevc_mdcv-p-ITU-R\ BT.2020_mdcv-l-10000_clli-10000.mp4

# -------------------------------------------------------------------------------

# AV1
MP4Box -new \
  -add hdr_media/av1_mdcv-p-ITU-R\ BT.2020_mdcv-l-10000_clli-10000.obu:fmt=obu:fps=24 \
  ./hdr_media/av1_mdcv-p-ITU-R\ BT.2020_mdcv-l-10000_clli-10000.mp4
```

1点だけ引数の補足説明をしておく。

* fps=24 を付けたのは、規格上 H.265/AV1 の bitstream はフレームレート情報を含めなくても成立するためである ((今回の検証ではフレームレートの確認はしないので、本当に念の為に加えた引数である))


#### 5.5. AVIF

##### 5.5.1. 概要

AVIF の作成は下図のように libavif のビルド時に生成される avifenc コマンドを使って行った。なお、図を見て分かるように MDCV の情報は付与していない。これは libavif 側で [MDCV の情報を埋め込む実装が無かったから](https://github.com/AOMediaCodec/libavif/blob/v1.3.0/src/write.c#L723) である。ご了承いただきたい。

<figure class="figure-image figure-image-fotolife" title="図xx. AVIFファイル作成手順">[f:id:takuver4:20260219070405p:plain:w350]<figcaption>図xx. AVIFファイル作成手順</figcaption></figure>

##### 5.5.2. コマンドライン引数

avifenc コマンドで使用したコマンドライン引数を以下に示す。

```bash
avifenc \
  ./src_img/1920x1080_ST2084_Rec.2020.png \
  -d 10 \
  --cicp 9/16/0 \
  -c aom \
  --lossless \
  --clli 10000,10000 \
  --ignore-exif \
  ./hdr_media/avif_mdcv-p-None_mdcv-l-None_clli-10000.avif
```

いくつかの引数について、以下に箇条書きで補足説明をしておく。

* `--cicp`の Matrix Coefficients が`0`なのは RGB エンコードを指定したため
* `-c aom` としてコーデックを libaom にしたのは RGB でエンコードを行うため
  * 余談だが SVT-AV1 は YCbCr 形式にしか対応してなかった
* `--ignore-exif`はワーニング表示を消すため（これは筆者環境の問題なのか…？）

#### 5.6. PNG

##### 5.6.1. 概要

CICP、MDCV、CLLI の情報を持つ PNG ファイルは下図のように 2段階で FFmpeg コマンドを叩いて作成した。
理由は PNG に MDCV、CLLI を書き込めるツールは FFmpeg だけだったからである。

<figure class="figure-image figure-image-fotolife" title="図xx. PNG ファイル作成手順">[f:id:takuver4:20260301102831p:plain:w650]<figcaption>図xx. PNG ファイル作成手順</figcaption></figure>

HEVC、AV1 で説明したように MDCV、CLLI を書き込むには side_data に適切にデータを入れる必要があるのだが、
一度 bitstream を作ってから PNG に変換した場合は上手く行ったのでこの方式を取った ((改めて考えると、もう少し工夫すれば HEVC と AV1 も同じ手が使えたのかもしれない…))。

ただし、ブログ作成時点の FFmpeg の libavcodec/pngenc.c には MDCV のアドレス計算ミスがあったので、[ローカルで修正したもの](https://github.com/toru-ver4/FFmpeg_png_mdcv/commit/eb78d47428cbae85f6e01d702ca69802764ebdd2) をビルドして使用した。

##### 5.6.2. コマンドライン引数

メタデータ付きの PNG 作成に使用したコマンドライン引数は以下の通り。初めに以下の通りに bitstream を作成した。

```bash
# Bitstream (HEVC)
ffmpeg \
  -hide_banner \
  -loop 1 \
  -color_primaries bt2020 \
  -color_trc smpte2084 \
  -colorspace bt2020nc \
  -framerate 24 \
  -i ./src_img/1920x1080_ST2084_Rec.2020.png \
  -frames:v 1 \
  -c:v libx265 \
  -x265-params \
    colorprim=bt2020:\
    transfer=smpte2084:\
    colormatrix=bt2020nc:\
    range=limited:\
    master-display=G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(100000000,0):\
    max-cll=10000,10000 \
  -color_primaries bt2020 \
  -color_trc smpte2084 \
  -colorspace bt2020nc \
  -pix_fmt yuv444p12le \
  -qp 0 \
  -f hevc \
  hdr_media/png_mdcv-p-ITU-R\ BT.2020_mdcv-l-10000_clli-10000.h265
```

続けて bitstream からメタデータ付きの PNG を生成した。

```bash
# PNG
ffmpeg \
  -hide_banner \
  -f hevc \
  -i hdr_media/png_mdcv-p-ITU-R\ BT.2020_mdcv-l-10000_clli-10000.h265 \
  -frames:v 1 \
  -update 1 \
  ./hdr_media/png_mdcv-p-ITU-R\ BT.2020_mdcv-l-10000_clli-10000.png
```

一部の引数について、箇条書きで補足説明をしておく。

* 対象が静止画の PNG だったので `-pix_fmt yuv444p12le` という 4:4:4 の設定を使用した
  * ただし HEVC の Full Range には [不安があった](https://trev16.hatenablog.com/entry/2025/03/20/155546) ので `-x265-params` には `range=limited` を設定した

# 6. 詳細 (メタデータ確認)

ここでは、作成したファイルのメタデータの確認方法を述べる。最初に大まかな方針を述べ、その後に具体的な手順を説明する。

#### 6.1. 方針

作成した HEVC、AV1、AVIF、PNG ファイルのメタデータが表3 の通りとなっているか確認した。
筆者にはバイナリのパーサーを作る能力は無いので、[gpac](https://gpac.io/)、[MP4Box](https://gpac.io/)、[pngcheck](https://github.com/pnggroup/pngcheck) らのツールを併用してメタデータをテキストとして出力し、その値を確認する方針を取った。

以下で、HEVC、AV1、AVIF、PNG の各種フォーマットでの具体的な確認方法を述べていく。

#### 6.2. HEVC

##### 6.2.1. bitstream

HEVC の bitstream は gpac コマンドを使って XML に変換した後、[dasel](https://github.com/TomWright/dasel) を使って JSON 変換した。XML から JSON に変換した理由は 「JSON の方が要素ごとに改行が入って見やすかった」という割とどうでもいいものである。通常は XML で良いと考える。

使用したコマンドの具体例を以下に示す。

```bash
# bitstream -> XML
gpac \
  -i "./hdr_media/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.h265" \
  inspect:deep:analyze=on \
  > "./data/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.h265.xml"

# XML -> JSON
dasel \
  -f "./data/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.h265.xml" \
  -r xml \
  -w json \
  > "./data/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.h265.json"
```

<br>

JSON に変換した後は、以下の表に示す値が期待値通りか一つずつ確認するテストコードを書いた。

<div style="text-align: center; margin: 1.5em 0;">
  <div style="font-weight: bold; margin-bottom: 0.5em;">
    表3. HDR メタデータ指定時の各オプションと確認内容
  </div>
  <table style="margin-left: auto; margin-right: auto; border-collapse: collapse;">
    <thead>
      <tr>
        <th>項目</th>
        <th>確認内容</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>-colour_primaries</td>
        <td> 9 であること</td>
      </tr>
      <tr>
        <td>-transfer_characteristic</td>
        <td> 16 であること</td>
      </tr>
      <tr>
        <td>-matrix_coeffs</td>
        <td> 9 であること</td>
      </tr>
      <tr>
        <td>-video_full_range_flag</td>
        <td>Limited Range なので 0 であること</td>
      </tr>
      <tr>
        <td>-display_primaries_x</td>
        <td>
          設定した x 色度を 0.00002 で割った整数値であること、<br>
          G, B, R の順であること (( 詳細は ITU-T H.265 (V10) (07/2024) の D.3.28 Mastering display colour volume SEI message semantics の項目を参照 ))
        </td>
      </tr>
      <tr>
        <td>-display_primaries_y</td>
        <td>
          設定した y 色度を 0.00002 で割った整数値であること、<br>
          G, B, R の順であること
        </td>
      </tr>
      <tr>
        <td>-white_point_x</td>
        <td>設定した x 色度を 0.00002 で割った整数値であること</td>
      </tr>
      <tr>
        <td>-white_point_y</td>
        <td>設定した y 色度を 0.00002 で割った整数値であること</td>
      </tr>
      <tr>
        <td>-max_display_mastering_luminance</td>
        <td>設定した輝度 (cd/㎡) を 0.0001 で割った整数値であること (( 詳細は ITU-T H.265 (V10) の D.3.28 Mastering display colour volume SEI message semantics の項目を参照 ))</td>
      </tr>
      <tr>
        <td>-min_display_mastering_luminance</td>
        <td>0 であること（今回は 0 固定としたため）</td>
      </tr>
      <tr>
        <td>-max_content_light_level</td>
        <td>設定した輝度 (cd/㎡) を示す整数値であること (( 詳細は ITU-T H.265 (V10) の D.3.35 Content light level information SEI message semantics の項目を参照 ))</td>
      </tr>
      <tr>
        <td>-max_pic_average_light_level</td>
        <td>設定した輝度 (cd/㎡) を示す整数値であること</td>
      </tr>
    </tbody>
  </table>
</div>

##### 6.2.2. MP4 コンテナ

HEVC の MP4 コンテナは MP4Box コマンドを使って XML に変換した後、 [dasel](https://github.com/TomWright/dasel) を使って JSON 変換した。XML から JSON に変換した理由は 「JSON の方が要素ごとに改行が入って見やすかった」という割とどうでもいいものである。通常は XML で良いと考える。

使用したコマンドの具体例を以下に示す。

```bash
# MP4 -> XML
MP4Box \
  -stdb \
  -dxml \
  "./hdr_media/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.mp4" \
  > "./data/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.mp4.xml"

# XML -> JSON
dasel \
  -f "./data/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.mp4.xml" \
  -r xml \
  -w json \
  > "./data/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.mp4.json"
```

<br>

JSON に変換した後は、メタデータが以下の表の通りか一つずつ確認するテストコードを書いた。なお、確認内容に「要出典」の文字列があるように、筆者が金銭的な問題で [ISO/IEC 14496-12](https://www.iso.org/standard/83102.html) を購入できていないため、期待値が本当にこの値で正しいことの裏取りは取れていない(( MP4Box 以外の FFmpeg の実装を ChatGPT 先生に調べてもらったところ、この値だったので間違っている可能性は低いと筆者は考えている)) 。

<div style="text-align: center; margin: 1.5em 0;">
  <div style="font-weight: bold; margin-bottom: 0.5em;">
    表4. HDR メタデータ指定時の各オプションと確認内容（色度割当明示版）
  </div>
  <table style="margin-left: auto; margin-right: auto; border-collapse: collapse;">
    <thead>
      <tr>
        <th>項目</th>
        <th>確認内容</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>-colour_primaries</td>
        <td>9 であること</td>
      </tr>
      <tr>
        <td>-transfer_characteristic</td>
        <td> 16 であること</td>
      </tr>
      <tr>
        <td>-matrix_coeffs</td>
        <td> 9 であること</td>
      </tr>
      <tr>
        <td>-video_full_range_flag</td>
        <td>Limited Range なので 0 であること</td>
      </tr>
      <tr>
        <td>-display_primaries_0_x</td>
        <td>
          設定した Green の x 色度を 0.00002 で割った整数値であること&lt;要出典&gt;
        </td>
      </tr>
      <tr>
        <td>-display_primaries_0_y</td>
        <td>
          設定した Green の y 色度を 0.00002 で割った整数値であること&lt;要出典&gt;
        </td>
      </tr>
      <tr>
        <td>-display_primaries_1_x</td>
        <td>
          設定した Blue の x 色度を 0.00002 で割った整数値であること&lt;要出典&gt;
        </td>
      </tr>
      <tr>
        <td>-display_primaries_1_y</td>
        <td>
          設定した Blue の y 色度を 0.00002 で割った整数値であること&lt;要出典&gt;
        </td>
      </tr>
      <tr>
        <td>-display_primaries_2_x</td>
        <td>
          設定した Red の x 色度を 0.00002 で割った整数値であること&lt;要出典&gt;
        </td>
      </tr>
      <tr>
        <td>-display_primaries_2_y</td>
        <td>
          設定した Red の y 色度を 0.00002 で割った整数値であること&lt;要出典&gt;
        </td>
      </tr>
      <tr>
        <td>-white_point_x</td>
        <td>
          設定した x 色度を 0.00002 で割った整数値であること&lt;要出典&gt;
        </td>
      </tr>
      <tr>
        <td>-white_point_y</td>
        <td>
          設定した y 色度を 0.00002 で割った整数値であること&lt;要出典&gt;
        </td>
      </tr>
      <tr>
        <td>-max_display_mastering_luminance</td>
        <td>
          設定した輝度 (cd/㎡) を 0.0001 で割った整数値であること&lt;要出典&gt;
        </td>
      </tr>
      <tr>
        <td>-min_display_mastering_luminance</td>
        <td>0 であること（今回は 0 固定としたため）</td>
      </tr>
      <tr>
        <td>-max_content_light_level</td>
        <td>
          設定した輝度 (cd/㎡) を示す整数値であること&lt;要出典&gt;
        </td>
      </tr>
      <tr>
        <td>-max_pic_average_light_level</td>
        <td>
          設定した輝度 (cd/㎡) を示す整数値であること&lt;要出典&gt;
        </td>
      </tr>
    </tbody>
  </table>
</div>

#### 6.3. AV1

##### 6.3.1. bitstream

AV1 の bitstream も HEVC と同様に gpac コマンドを使って XML に変換した後、 [dasel](https://github.com/TomWright/dasel) を使って JSON 変換した。

使用したコマンドの具体例を以下に示す。

```bash
# bitstream -> XML
gpac \
  -i "./hdr_media/av1_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.obu" \
  inspect:deep:analyze=on \
  > "./data/av1_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.obu.xml"

# XML -> JSON
dasel \
  -f "./data/av1_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.obu.xml" \
  -r xml \
  -w json \
  > "./data/av1_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.obu.json"
```

<br>

JSON に変換した後は、メタデータが以下の表の通りか一つずつ確認するテストコードを書いた。

<div style="text-align: center; margin: 1.5em 0;">
  <div style="font-weight: bold; margin-bottom: 0.5em;">
    表5. AV1 / gpac における HDR メタデータ項目と確認内容
  </div>
  <table style="margin-left: auto; margin-right: auto; border-collapse: collapse;">
    <thead>
      <tr>
        <th>項目</th>
        <th>確認内容</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>-color_primaries</td>
        <td>9 であること</td>
      </tr>
      <tr>
        <td>-transfer_characteristics</td>
        <td>16 であること</td>
      </tr>
      <tr>
        <td>-matrix_coefficients</td>
        <td>9 であること</td>
      </tr>
      <tr>
        <td>-color_range</td>
        <td>Limited Range なので 0 であること</td>
      </tr>
      <tr>
        <td>-display_primaries_x ((AV1 の正式な文言は "primary_chromaticity_x" なのだが、gpac が出力時に "display_primaries_x" としたため、テストコードもこの文字列を使った https://github.com/gpac/gpac/blob/v26.02.0/src/filters/inspect.c#L501-L529))</td>
        <td>
          設定した x 色度を 1/65535 で割った整数値であること、R, G, B の順であること
          ((詳細は AV1 Bitstream &amp; Decoding Process Specification (Version 1.0.0 with Errata 1) の 6.7.4. Metadata high dynamic range mastering display color volume semantics の項目を参照))
        </td>
      </tr>
      <tr>
        <td>-display_primaries_y ((AV1 の正式な文言は "primary_chromaticity_y" なのだが、gpac が出力時に "display_primaries_y" としたため、テストコードもこの文字列を使った https://github.com/gpac/gpac/blob/v26.02.0/src/filters/inspect.c#L501-L529))</td>
        <td>設定した y 色度を 1/65535 で割った整数値であること、R, G, B の順であること</td>
      </tr>
      <tr>
        <td>-white_point_x ((AV1 の正式な文言は "white_point_chromaticity_x" なのだが、gpac が出力時に "white_point_x" としたため、テストコードもこの文字列を使った https://github.com/gpac/gpac/blob/v26.02.0/src/filters/inspect.c#L501-L529))</td>
        <td>設定した x 色度を 1/65535 で割った整数値であること</td>
      </tr>
      <tr>
        <td>-white_point_y ((AV1 の正式な文言は "white_point_chromaticity_y" なのだが、gpac が出力時に "white_point_y" としたため、テストコードもこの文字列を使った https://github.com/gpac/gpac/blob/v26.02.0/src/filters/inspect.c#L501-L529))</td>
        <td>設定した y 色度を 1/65535 で割った整数値であること</td>
      </tr>
      <tr>
        <td>-max_display_mastering_luminance ((AV1 の正式な文言は "luminance_max" なのだが、gpac が出力時に "max_display_mastering_luminance" としたため、テストコードもこの文字列を使った https://github.com/gpac/gpac/blob/v26.02.0/src/filters/inspect.c#L501-L529))</td>
        <td>
          設定した輝度 (cd/㎡) を示す数値であること
          ((AV1 の規格では 24-bit 整数、8-bit 小数の固定小数点フォーマットなのだが、gpac がダンプ時に cd/㎡ の単位に変換しているので、今回は cd/㎡ 単位で比較した https://github.com/gpac/gpac/blob/v26.02.0/src/filters/inspect.c#L501-L529))
        </td>
      </tr>
      <tr>
        <td>-min_display_mastering_luminance ((AV1 の正式な文言は "luminance_min" なのだが、gpac が出力時に "min_display_mastering_luminance" としたため、テストコードもこの文字列を使った https://github.com/gpac/gpac/blob/v26.02.0/src/filters/inspect.c#L501-L529))</td>
        <td>0 であること（今回は 0 固定としたため）</td>
      </tr>
      <tr>
        <td>-max_content_light_level ((AV1 の正式な文言は "luminance_max" なのだが、gpac が出力時に "max_content_light_level" としたため、テストコードもこの文字列を使った https://github.com/gpac/gpac/blob/v26.02.0/src/filters/inspect.c#L501-L529))</td>
        <td>設定した輝度 (cd/㎡) を示す整数値であること</td>
      </tr>
      <tr>
        <td>-max_pic_average_light_level ((AV1 の正式な文言は "max_fall" なのだが、gpac が出力時に "max_pic_average_light_level" としたため、テストコードもこの文字列を使った https://github.com/gpac/gpac/blob/v26.02.0/src/filters/inspect.c#L501-L529))</td>
        <td>設定した輝度 (cd/㎡) を示す整数値であること</td>
      </tr>
    </tbody>
  </table>
</div>

##### 6.3.2. MP4 コンテナ

AV1 の MP4 コンテナも HEVC と同様に MP4Box コマンドを使って XML に変換した後、 [dasel](https://github.com/TomWright/dasel) を使って JSON 変換した。

使用したコマンドの具体例を以下に示す。

```bash
# MP4 -> XML
MP4Box \
  -stdb \
  -dxml \
  "./hdr_media/av1_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.mp4" \
  > "./data/av1_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.mp4.xml"

# XML -> JSON
dasel \
  -f "./data/av1_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.mp4.xml" \
  -r xml \
  -w json \
  > "./data/av1_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.mp4.json"
```

<br>

JSON に変換した後は、メタデータが以下の表の通りか一つずつ確認するテストコードを書いた。なお、AV1 の MP4 コンテナ確認に関しては補足が3点ある。

1点目、MP4 コンテナでのデータの持ち方は AV1 も HEVC も同じなので&lt;要出典&gt;、確認内容は原則として HEVC の時と同じである。

2点目、確認事項は HEVC と同じなのだが、AV1 と HEVC とでは <span style="color: #ff5252">bitstream での情報の持ち方</span>に差異がある。例えば xy 色度は HEVC が 0.00002、AV1 は (1/65535) で割った値を保持している。こういったデータ形式の差に起因する量子化誤差の発生はテストの際に許容している。

3点目、先ほどの HEVC と AV1 の bitstream のデータ形式の違いは、MP4 コンテナ変換時に [gf_av1_format_mdcv_to_mpeg](https://github.com/gpac/gpac/blob/v26.02.0/src/media_tools/av_parsers.c#L2384-L2411) という関数で処理されるのだが、RGB to GBR のデータ並べ替えのコードに誤りがあった。そのため、今回のテストでは [筆者のローカルで修正した](https://github.com/toru-ver4/gpac/commit/0df998e4ad40833d64d256462c34e82867a33917#diff-85593a2543dbc09dce05fbe45b933dcc7a837880b482bd12858c4497ae796314) MP4Box をビルドして使用した。

<div style="text-align: center; margin: 1.5em 0;">
  <div style="font-weight: bold; margin-bottom: 0.5em;">
    表6. HDR メタデータ指定時の各オプションと確認内容（色成分別指定）
  </div>
  <table style="margin-left: auto; margin-right: auto; border-collapse: collapse;">
    <thead>
      <tr>
        <th>項目</th>
        <th>確認内容</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>-color_primaries</td>
        <td>9 であること</td>
      </tr>
      <tr>
        <td>-transfer_characteristics</td>
        <td>16 であること</td>
      </tr>
      <tr>
        <td>-matrix_coefficients</td>
        <td>9 であること</td>
      </tr>
      <tr>
        <td>-color_range</td>
        <td>Limited Range なので 0 であること</td>
      </tr>
      <tr>
        <td>-display_primaries_0_x</td>
        <td>設定した Green の x 色度を 0.00002 で割った整数値であること&lt;要出典&gt;</td>
      </tr>
      <tr>
        <td>-display_primaries_0_y</td>
        <td>設定した Green の y 色度を 0.00002 で割った整数値であること&lt;要出典&gt;</td>
      </tr>
      <tr>
        <td>-display_primaries_1_x</td>
        <td>設定した Blue の x 色度を 0.00002 で割った整数値であること&lt;要出典&gt;</td>
      </tr>
      <tr>
        <td>-display_primaries_1_y</td>
        <td>設定した Blue の y 色度を 0.00002 で割った整数値であること&lt;要出典&gt;</td>
      </tr>
      <tr>
        <td>-display_primaries_2_x</td>
        <td>設定した Red の x 色度を 0.00002 で割った整数値であること&lt;要出典&gt;</td>
      </tr>
      <tr>
        <td>-display_primaries_2_y</td>
        <td>設定した Red の y 色度を 0.00002 で割った整数値であること&lt;要出典&gt;</td>
      </tr>
      <tr>
        <td>-white_point_x</td>
        <td>設定した x 色度を 0.00002 で割った整数値であること&lt;要出典&gt;</td>
      </tr>
      <tr>
        <td>-white_point_y</td>
        <td>設定した y 色度を 0.00002 で割った整数値であること&lt;要出典&gt;</td>
      </tr>
      <tr>
        <td>-max_display_mastering_luminance</td>
        <td>設定した輝度 (cd/㎡) を 0.0001 で割った整数値であること&lt;要出典&gt;</td>
      </tr>
      <tr>
        <td>-min_display_mastering_luminance</td>
        <td>0 であること（今回は 0 固定としたため）</td>
      </tr>
      <tr>
        <td>-max_content_light_level</td>
        <td>設定した輝度 (cd/㎡) を示す整数値であること&lt;要出典&gt;</td>
      </tr>
      <tr>
        <td>-max_pic_average_light_level</td>
        <td>設定した輝度 (cd/㎡) を示す整数値であること&lt;要出典&gt;</td>
      </tr>
    </tbody>
  </table>
</div>

#### 6.4. AVIF

AVIF も HEVC や AV1 と同じく bitstream と AVIF コンテナの両方の確認を行った。ただし、avifenc コマンドを使った AVIF 作成では bitstream が自動的には生成されないので、FFmpeg を使って AVIF から bitstream を抽出する処理を別途行った。詳細を以下で述べていく。

##### 6.4.1. bitstream

AVIF から FFmpeg を使い bitstream を抽出し、その後に gpac コマンドを使って XML に変換と [dasel](https://github.com/TomWright/dasel) を使った JSON 変換を行った。

使用したコマンドの具体例を以下に示す。

```bash
# extract bitstream
ffmpeg \
  -hide_banner \
  -i "./hdr_media/avif_mdcv-p-None_mdcv-l-None_clli-10000.avif" \
  -map 0:v:0 \
  -c copy \
  -f obu \
  "./hdr_media/avif_mdcv-p-None_mdcv-l-None_clli-10000.obu" \
  -y

# bitstream to XML
gpac \
  -i "./hdr_media/avif_mdcv-p-None_mdcv-l-None_clli-10000.obu" \
  inspect:deep:analyze=on \
  > "./data/avif_mdcv-p-None_mdcv-l-None_clli-10000.obu.xml"

# XML to JSON
dasel \
  -f "./data/avif_mdcv-p-None_mdcv-l-None_clli-10000.obu.xml" \
  -r xml \
  -w json \
  > "./data/avif_mdcv-p-None_mdcv-l-None_clli-10000.obu.json"
```

<br>

JSON に変換した後は、メタデータが以下の表の通りか一つずつ確認するテストコードを書いた。なお、コンテナ側の確認に関しては 2点の特記事項がある。

1点目、Video Full Range Flag を除く CICP の値は全て 2 (Unspecified) となっている。これは libavif の仕様である。[ソースコードに記載されているコメント](https://github.com/AOMediaCodec/libavif/blob/v1.3.0/src/codec_aom.c#L868-L875) を要約すると「CICP を 2/2/2（Unspecified）にするのは、ISOBMFF の colr/nclx と整合性を保ちつつ、色記述を省略してビットを節約するため」である。

2点目、MDCV と CLLI に相当する情報は bitstream には埋め込まれない。MDCV に関しては [ソースコードで TODO 扱い](https://github.com/AOMediaCodec/libavif/blob/v1.3.0/src/write.c#L723) になっていたので納得はできる。一方で CLLI に関しては特に何も記述が見つからなかった。が、ソースコードを確認しても実装が見当たらなかったので、テストとしては「存在しないこと」を確認することにした。

<div style="text-align: center; margin: 1.5em 0;">
  <div style="font-weight: bold; margin-bottom: 0.5em;">
    表7. メタデータ未指定（Unspecified）時の各オプション条件
  </div>
  <table style="margin-left: auto; margin-right: auto; border-collapse: collapse;">
    <thead>
      <tr>
        <th>項目</th>
        <th>確認内容</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>-color_primaries</td>
        <td>2 (Unspecified) であること</td>
      </tr>
      <tr>
        <td>-transfer_characteristics</td>
        <td>2 (Unspecified) であること</td>
      </tr>
      <tr>
        <td>-matrix_coefficients</td>
        <td>2 (Unspecified) であること</td>
      </tr>
      <tr>
        <td>-color_range</td>
        <td>Full Range なので 1 であること</td>
      </tr>
      <tr>
        <td>-display_primaries_x</td>
        <td>存在しないこと</td>
      </tr>
      <tr>
        <td>-display_primaries_y</td>
        <td>存在しないこと</td>
      </tr>
      <tr>
        <td>-white_point_x</td>
        <td>存在しないこと</td>
      </tr>
      <tr>
        <td>-white_point_y</td>
        <td>存在しないこと</td>
      </tr>
      <tr>
        <td>-max_display_mastering_luminance</td>
        <td>存在しないこと</td>
      </tr>
      <tr>
        <td>-min_display_mastering_luminance</td>
        <td>存在しないこと</td>
      </tr>
      <tr>
        <td>-max_content_light_level</td>
        <td>存在しないこと</td>
      </tr>
      <tr>
        <td>-max_pic_average_light_level</td>
        <td>存在しないこと</td>
      </tr>
    </tbody>
  </table>
</div>

##### 6.4.2. AVIF コンテナ

これまでと同様に MP4Box コマンドを使って XML に変換した後、 [dasel](https://github.com/TomWright/dasel) を使って JSON 変換した。

使用したコマンドの具体例を以下に示す。

```bash
# AVIF → XML
MP4Box \
  -stdb \
  -dxml "./hdr_media/avif_mdcv-p-None_mdcv-l-None_clli-10000.avif" \
  > "./data/avif_mdcv-p-None_mdcv-l-None_clli-10000.avif.xml"

# XML → JSON
dasel \
  -f "./data/avif_mdcv-p-None_mdcv-l-None_clli-10000.avif.xml" \
  -r xml \
  -w json \
  > "./data/avif_mdcv-p-None_mdcv-l-None_clli-10000.avif.json"
```

JSON に変換した後は、メタデータが以下の表の通りか一つずつ確認するテストコードを書いた。なお、bitstream の比較に関しては 2点の特記事項がある。

1点目、AVIF コンテナ側には CICP 情報が正しく埋め込まれている。
2点目、MDCV 情報は bitstream と同様にコンテナにも埋め込まれない。CLLI のみが埋め込まれる。

<div style="text-align: center; margin: 1.5em 0;">
  <div style="font-weight: bold; margin-bottom: 0.5em;">
    表8. CICP 指定あり・MDCV 省略時の各オプション条件
  </div>
  <table style="margin-left: auto; margin-right: auto; border-collapse: collapse;">
    <thead>
      <tr>
        <th>項目</th>
        <th>確認内容</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>-color_primaries</td>
        <td>9 であること</td>
      </tr>
      <tr>
        <td>-transfer_characteristics</td>
        <td>16 であること</td>
      </tr>
      <tr>
        <td>-matrix_coefficients</td>
        <td>9 であること</td>
      </tr>
      <tr>
        <td>-color_range</td>
        <td>Full Range なので 1 であること</td>
      </tr>
      <tr>
        <td>-display_primaries_0_x</td>
        <td>存在しないこと</td>
      </tr>
      <tr>
        <td>-display_primaries_0_y</td>
        <td>存在しないこと</td>
      </tr>
      <tr>
        <td>-display_primaries_1_x</td>
        <td>存在しないこと</td>
      </tr>
      <tr>
        <td>-display_primaries_1_y</td>
        <td>存在しないこと</td>
      </tr>
      <tr>
        <td>-display_primaries_2_x</td>
        <td>存在しないこと</td>
      </tr>
      <tr>
        <td>-display_primaries_2_y</td>
        <td>存在しないこと</td>
      </tr>
      <tr>
        <td>-white_point_x</td>
        <td>存在しないこと</td>
      </tr>
      <tr>
        <td>-white_point_y</td>
        <td>存在しないこと</td>
      </tr>
      <tr>
        <td>-max_display_mastering_luminance</td>
        <td>存在しないこと</td>
      </tr>
      <tr>
        <td>-min_display_mastering_luminance</td>
        <td>存在しないこと</td>
      </tr>
      <tr>
        <td>-max_content_light_level</td>
        <td>設定した輝度 (cd/㎡) を示す整数値であること&lt;要出典&gt;</td>
      </tr>
      <tr>
        <td>-max_pic_average_light_level</td>
        <td>設定した輝度 (cd/㎡) を示す整数値であること&lt;要出典&gt;</td>
      </tr>
    </tbody>
  </table>
</div>

<br>

#### 6.5. PNG

PNG は [pngcheck](https://github.com/pnggroup/pngcheck) というコマンドを使って chunk の情報をテキストとして出力し、それを解析する方針とした。

使用したコマンドの具体例を以下に示す。

```bash
pngcheck \
  -v \
  "./hdr_media/png_mdcv-p-None_mdcv-l-None_clli-10000.png" \
  > "./data/png_mdcv-p-None_mdcv-l-None_clli-10000.png.txt"
```

テキストに変換した後は、メタデータが以下の表の通りか一つずつ確認するテストコードを書いた。

<div style="text-align: center; margin: 1.5em 0;">
  <div style="font-weight: bold; margin-bottom: 0.5em;">
    表9. cICP / MDCV / CLLI による HDR メタデータの意味付け
  </div>
  <table style="margin-left: auto; margin-right: auto; border-collapse: collapse;">
    <thead>
      <tr>
        <th>項目</th>
        <th>確認内容</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>cICP (eotf) ((筆者の都合で eotf と書いたが colorimetry の方が適切かも？))</td>
        <td>ITU-R BT.2100-PQ を示す文字列であること</td>
      </tr>
      <tr>
        <td>cICP (chromaticity)</td>
        <td>WRGB の xy 色度が BT.2020 であること</td>
      </tr>
      <tr>
        <td>cICP (range)</td>
        <td>"Full range" であること</td>
      </tr>
      <tr>
        <td>mDCV (chromaticity)</td>
        <td>WRGB の xy 色度が設定した値であること</td>
      </tr>
      <tr>
        <td>mDCV (luminance)</td>
        <td>Maximum luminance は指定した輝度 (cd/㎡)、Minimum luminance は 0 (cd/㎡) であること</td>
      </tr>
      <tr>
        <td>cLLI</td>
        <td>Maximum content light level と Maximum frame average light level の双方が指定した輝度 (cd/㎡) であること</td>
      </tr>
    </tbody>
  </table>
</div>

# 7. 感想

ブログを書くのに凄く時間がかかった。こういう自分の勉強を兼ねた技術調査って生成AI に丸投げするのは難しいなぁ、と思いながら記事を書いていた。

ソースコードの読解や規格文書の精読などは生成AI が大活躍してくれたので、もっと活用例を増やしていきたいところである。

# 8. 参考資料

* Recommendation ITU-T H.274 (V3), "Versatile supplemental enhancement information messages for coded video bitstreams", https://www.itu.int/rec/T-REC-H.274-202309-S/en
* gpac/gpac, "MP4Box Wiki", https://github.com/gpac/gpac/wiki/MP4Box 
* x265 documentation, "Command Line Options", https://x265.readthedocs.io/en/master/cli.html 
* SVT-AV1, "SVT-AV1 Parameters", https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/master/Docs/Parameters.md
* Recommendation ITU-T H.265 (V10) (07/2024), "High efficiency video coding", https://www.itu.int/rec/T-REC-H.265-202407-S
* AOMediaCodec/av1-spec, "AV1 Bitstream & Decoding Process Specification (Version 1.0.0 with Errata 1)", https://github.com/AOMediaCodec/av1-spec/releases/tag/v1.0.0-errata1
