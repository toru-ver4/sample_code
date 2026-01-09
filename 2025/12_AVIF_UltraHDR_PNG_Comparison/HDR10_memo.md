# HDR10 メモ

## HDR 10 について

* CEA (現CTA) が 2015年に発表
* 発表内容は HDR compatible video display であった
* HDR compatible video display の最小要件は以下
  * CEA-861-F（CEA-861.3 によって拡張）で定義された HDR 信号をサポートする少なくとも 1 つのインターフェースを備えていること。
  * 非圧縮映像用に、CEA-861.3 に準拠した静的 HDR メタデータを受信・処理できること。
  * IP、HDMI、またはその他の映像配信ソースから HDR10 メディアプロファイル* を受信・処理できること。加えて、その他のメディアプロファイルもサポートされる場合がある。
  * 画像を表示する前に、適切な Electro-Optical Transfer Function (EOTF：電気光学変換関数) を適用すること。
* ここで HDR10 Media Profile というものが出てくる
* で、その中にメタデータとして SMPTE ST 2086 と MaxFALL, MaxCLL が定義されてる

## SMPTE ST 2086 について

* 目的は To improve the color reproduction of this mastered content when shown on other displays
* ただし、This standard does not cover the color and tonal transformations that would be used to convert between different display color volumes.

| Item | Meaning |
|:-------|:-------|
| Display Primarids | Mastering display の primaries |
| Chromaticity of White Point | mastering display の white point |
| Maximum Display Mastering Luminance | mastering display のピーク輝度 |
| Minimum Display Mastering Luminance | mastering display の最小輝度 |

## EDID の HDR Static Metadata Data Block について

* 以下の 3つが含まれる
  * Desired Content Max Luminance data
  * Desired Content Max Frame-average Luminance data
  * Desired Content Min Luminance data
* このうち、Desired Content Max Frame-average Luminance data は FALL の Sink版だと考える

## PNG のメタデータを整理しましょう

* MDCV
  * Mastering display color primary chromaticities
  * Mastering display whote point chromaticity
  * Mastering display maximum luminance
  * Mastering display minimum luminance
* CLLI
  * Maximum Content Light Level (MaxCLL)
  * Maximum Frame-Average Light Level (MaxFALL)
