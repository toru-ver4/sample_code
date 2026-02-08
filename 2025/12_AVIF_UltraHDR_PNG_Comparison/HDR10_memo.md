# HDR10 メモ

## HDR 10 について

* CEA (現CTA) が [2015年に発表](https://web.archive.org/web/20170113053600/https://www.cta.tech/News/Press-Releases/2015/August/CEA-Defines-%E2%80%98HDR-Compatible%E2%80%99-Displays.aspx)
* 発表内容は HDR compatible video display であった
* HDR compatible video display の最小要件は以下
  * CEA-861-F（CEA-861.3 によって拡張）で定義された HDR 信号をサポートする少なくとも 1 つのインターフェースを備えていること。
  * 非圧縮映像用に、CEA-861.3 に準拠した静的 HDR メタデータを受信・処理できること。
  * IP、HDMI、またはその他の映像配信ソースから HDR10 メディアプロファイルを受信・処理できること。加えて、その他のメディアプロファイルもサポートされる場合がある。
  * 画像を表示する前に、適切な Electro-Optical Transfer Function (EOTF：電気光学変換関数) を適用すること。
* ここで HDR10 Media Profile というものが出てくる
* HDR Media Profile の定義は以下
  * EOTF: SMPTE ST 2084
  * Color Sub-sampling: 4:2:0 (for compressed video sources)
  * Bit Depth: 10 bit
  * Color Primaries:  ITU-R BT.2020
  * Metadata: SMPTE ST 2086, MaxFALL, MaxCLL
* Metadata の中に SMPTE ST 2086, MaxFALL, MaxCLL があることが分かる

### SMPTE ST 2086 について

* 目的は To improve the color reproduction of this mastered content when shown on other displays
* ただし、This standard does not cover the color and tonal transformations that would be used to convert between different display color volumes.

| Item | Meaning |
|:-------|:-------|
| Display Primarids | Mastering display の primaries |
| Chromaticity of White Point | mastering display の white point |
| Maximum Display Mastering Luminance | mastering display のピーク輝度 |
| Minimum Display Mastering Luminance | mastering display の最小輝度 |

## MDCV と CLLI　の定義について

以下の対応関係がある

| HDR10 での名称 | CTA-861-* | SMPTE | ISO/IEC |
|:-------:|:-------:|:-------:|:-------:|
| SMPTE ST 2086 | Dynamic Range and Mastering InfoFrame の前半4つ | SMPTE ST 2086 | ISO/IEC 14496-12 の MDCV<要裏どり> |
| MaxFALL, MaxCLL | Dynamic Range and Mastering InfoFrame の後半2つ | - | ISO/IEC 14496-12 の CLLI<要裏どり> |

## Source側の情報

### SMPTE St 2086

タイトルに "Mastering Display Color Volume Metadata Supporting High Luminance and Wide Color Gamut Images" が入っている前提で、以下が定義されている。

* Display Primaries
* Chromaticity of White Point
* Maximum Display Mastering Luminance
* Minimum Display Mastering Luminance

### CTA

Dynamic Range and Mastering InfoFrame に情報あり

* dispay_primaries
* white_point
* max_display_mastering_luminance
* min_display_mastering_luminance
* Maximum Content Light Level
* Maximum Frame-average Light Level

### PNG のメタデータを整理しましょう

* MDCV
  * Mastering display color primary chromaticities
  * Mastering display whote point chromaticity
  * Mastering display maximum luminance
  * Mastering display minimum luminance
* CLLI
  * Maximum Content Light Level (MaxCLL)
  * Maximum Frame-Average Light Level (MaxFALL)

### AVIF

* [AV1-ISOBMFF](https://aomediacodec.github.io/av1-isobmff/)
* [AVIF Spec](https://aomediacodec.github.io/av1-avif)

* mdcv と clli はスペックとして存在

## Sink側の情報

### EDID

* HDR Static Metadata Data Block の中に以下の 3つの指標がある
  * Desired Content Max Luminance data
    * This is the content’s absolute peak luminance (in cd/m2) (likely only in a small area of the screen) that the display prefers for optimal content rendering.
  * Desired Content Max Frame-average Luminance data
    * This is the content’s max frame-average luminance (in cd/m2) that the display prefers for optimal content rendering
  * Desired Content Min Luminance data
    * This is the minimum value of the content (in cd/m2) that the display prefers for optimal content rendering

## 確認する項目

* AVIF without Gain Map
* PNG
* av1
* hevc

### 除外したやつリスト

* AVIF with Gain Map
  * Gain Map が絡むと訳がわからなくなる
* JPEG XL
  * MDCV, CLLI 非サポート
* UltraHDR
  * MDCV, CLLI 非サポート
* HEIF
  * サポートされてるブラウザが少ない

### MP4 とか AV1 とか AVIF のあれ

* colr とかの格納場所が定義された文書

* ISO/IEC 14496-12	VisualSampleEntry 内に colr を含められると明記
* ISO/IEC 14496-15	AVC SampleEntry（avc1/avc3）に colr が optional と明示
* ISO/IEC 23008-15	HEVC SampleEntry（hvc1/hev1）で同様に optional と記述
* ISO/IEC 14496-30	AV1 SampleEntry（av01）にも optional box として指定

### HEVC bitstream

* SPS の VUI parameters 
  * colour_primaries
  * transfer_characteristics
  * matrix_coeffs
  * video_full_range_flag

* SEI の Mastering display colour volume
  * display_primaries_x
  * display_primaries_y
  * white_point_x
  * white_point_y
  * max_display_mastering_luminance
  * min_display_mastering_luminance

* SEI の Content light level information
  * max_content_light_level
  * max_pic_average_light_level

### AVI OBU

* Sequence header OBU の Color config
  * color_primaries
  * transfer_characteristics
  * matrix_coefficients
  * color_range

* Metadata OBU syntax の Metadata high dynamic range mastering display color volume
  * primary_chromaticity_x
  * primary_chromaticity_y
  * white_point_chromaticity_x
  * white_point_chromaticity_y
  * luminance_max
  * luminance_min

* Metadata OBU syntax の Metadata high dynamic range content light level
  * max_cll
  * max_fall


### libavif メモ

* MDCV に相当するデータは扱わないっぽい
* colr box は avifReadColorNclxProperty で読んでる
* "clli" はこの文字で検索すれば出てくる
* OBU のパースは以下で行ってる
  * avifdec.c の avifDecoderParse 
  * read.c の avifDecoderReset -> avifSequenceHeaderParse
  * ただし、avifSequenceHeaderParse は 'colr' ボックスから情報が取得できなかった場合のみ。つまり、colrボックスが存在していれば気にされない
* エンコード時、OBU には cicp情報は埋め込まれない
  * そのため、初期値の 2/2/2 のままになってる
  * https://gitlab.com/webmproject/libaom/-/blob/v3.13.1/av1/av1_cx_iface.c?ref_type=tags#L432-L434


### Test のメモ

* AV1 の OBU の primary_chromaticity_x は gpac でダンプすると display_primaries_x に名前が変わる
