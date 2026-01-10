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
* [AV1 Spec](https://aomediacodec.github.io/av1-avif)

* mdcv と clli はスペックとして存在

## Sink側の情報

### EDID の HDR Static Metadata Data Block について

当時の俺は、なんで EDID すなわちモニター側の情報を調べたんだろう。調べるべきは AVI Info Frame では…

* 以下の 3つが含まれる
  * Desired Content Max Luminance data
  * Desired Content Max Frame-average Luminance data
  * Desired Content Min Luminance data
* このうち、Desired Content Max Frame-average Luminance data は FALL の Sink版だと考える

