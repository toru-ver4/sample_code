# テスト画像作成

## 背景

Edge/Chrome で PNG / AVIF / HEIC / JPEG XL の HDRフォーマット画像を表示できるか確認する。
画像には今のところ以下の可能性がある。

| 画像フォーマット | CICP (画像埋め込み) | CLLI (画像埋め込み) | CICP (ICC Profile) |
|:----:|:----:|:----:|:----:|
| PNG | Supported (cICP chunk)[^1] | Supported (cLLI chunk)[^2] | Supported (iCCP chunk)[^3] |
| AVIF | Supported[^4] | Supported (cLLI chunk)[^5] | Supported (iCCP chunk)[^6] |
| HEIC | Supported[^7] | Supported (cLLI chunk)[^8] | Not Supported (技術的には可能なんだけど `heif-enc` でできない) |
| JPEG XL | Supported[^9] | Not Supported | Supported[^10] |

[^1]: `ty_lib/test_pattern_generator2.py` の `add_hdr_info_to_png` を使って埋め込める
[^2]: `ty_lib/test_pattern_generator2.py` の `add_hdr_info_to_png` を使って埋め込める
[^3]: `ty_lib/test_pattern_generator2.py` の `add_icc_profile_using_exiftool` を使って埋め込める
[^4]: `avifenc` コマンドの `--cicp` オプションで埋め込める
[^5]: `avifenc` コマンドの `--clli` オプションで埋め込める
[^6]: `avifenc` コマンドの `--icc` オプションで埋め込める
[^7]: `heif-enc` コマンドの `--matrix_coefficients`, `--colour_primaries`, `--transfer_characteristic`, `--full_range_flag` オプションで埋め込める
[^8]: `heif-enc` コマンドの `--clli` オプションで埋め込める
[^9]: `cjxl` コマンドの `-x color_space=RGB_D65_202_Rel_PeQ` オプションで埋め込み可能
[^10]: `cjxl` コマンドの `-x icc_pathname=icc_profile_path` オプションで埋め込み可能

## bit深度

2026/04_ICC_Profile_for_HDR_Media/prompt/add_icc_profile_using_exiftool.md に合わせる

## 画像作成

CICP = 9-16-0-1 (BT.2020-PQ, RGB444, Full Range), cLLI = (203 nits, 203 nits) として
`2019/012_colour_v0.3.14_check/img/SMPTE ST2084_ITU-R BT.2020_D65_1920x1080_rev07_type1.png` に対して
CICP (画像埋め込み), CLLI (画像埋め込み), CICP (ICC Profile) の組み合わせの総当りのファイルを作成して欲しいです。
情報が存在する、存在しないによって、Webブラウザやドローツールが HDR と認識するのかしないのかを簡単に判別したいです。

対象環境は、Windows 上で動作する比較的新しいバージョンの Edge および Chrome とする。

画像生成の Python スクリプトのファイル名は `2026/04_ICC_Profile_for_HDR_Media/make_test_images.py` とする

### 組み合わせ

各画像形式で対応している情報の有無を総当たりにし、以下の合計24画像を作成すること。

| 画像フォーマット | CICP (画像埋め込み) | CLLI (画像埋め込み) | CICP (ICC Profile) | 組み合わせ数 |
|:----:|:----:|:----:|:----:|:----:|
| PNG | 有／無 | 有／無 | 有／無 | 8 |
| AVIF | 有／無 | 有／無 | 有／無 | 8 |
| HEIC | 有／無 | 有／無 | 非対応のため常に無 | 4 |
| JPEG XL | 有／無 | 非対応のため常に無 | 有／無 | 4 |

CICP (画像埋め込み) と CICP (ICC Profile) の両方が「有」の組み合わせでは、両方の情報を同時に存在させること。

### ICC Profile

CICP = 9-16-0-1 に相当する BT.2020/PQ ICC Profile を新規生成して使用すること。
ICC Profile の生成は、`2026/04_ICC_Profile_for_HDR_Media/create_icc_profile.py` の
`create_bt2020_pq_curve_4096_with_cicp_profile` を参考にすること。

### 元画像のメタデータ

元画像に CICP、CLLI、ICC Profile など、今回の比較に影響する既存情報が含まれている場合は、画像データを変えずに削除してから各組み合わせを作成すること。

### エンコード条件

各形式のエンコード条件は以下の通りとする。

| 画像フォーマット | ビット深度 | エンコード | Chroma subsampling |
|:----:|:----:|:----:|:----:|
| PNG | 16-bit | lossless | RGB 4:4:4 |
| AVIF | 10-bit | lossless | RGB 4:4:4 |
| HEIC | 10-bit | lossless | RGB 4:4:4 |
| JPEG XL | 16-bit | lossless (`cjxl -q 100`) | RGB 4:4:4 |

ビット深度と各形式の変換方法については、
`2026/04_ICC_Profile_for_HDR_Media/prompt/add_icc_profile_using_exiftool.md` に合わせること。
lossless と品質値の両方を指定できない、または品質値が意味を持たないエンコーダーでは、lossless 指定を優先すること。

### ファイル名

ファイル名から画像形式と各情報の有無を判別できる命名規則にすること。
例えば、PNG で CICP が有、CLLI が無、ICC Profile が有の場合は
`png_cicp-on_clli-off_icc-on.png` とする。

画像は `2026/04_ICC_Profile_for_HDR_Media/test_img` に作成すること。
また、結果は https://toru-ver4.github.io/pages_test/MDCV_CLLI_Test/index.html を参考に
全画像をリンククリックで表示できるような HTML を作成すること。

HTML ファイルは `2026/04_ICC_Profile_for_HDR_Media/hdr_judge_check.html` という単体の HTML で作ること。

HTML は以下の要件を満たすこと。

* 画像形式別、メタデータの組み合わせ別の表にする
* 各リンクに CICP、CLLI、ICC Profile の有無を明記する
* 画像リンクは新規タブで開く
* ブラウザが対応している形式については、ページ内でも画像を確認できるようにする
* HTML 単体で動作するようにし、外部の CSS や JavaScript に依存しない

## 検証

`2026/04_ICC_Profile_for_HDR_Media/prompt/png_hdr_info_insertion_spec_for_codex.md` および `2026/04_ICC_Profile_for_HDR_Media/prompt/add_icc_profile_using_exiftool.md` の仕様書や既存実装を参考に、生成結果を検証すること。
少なくとも以下を確認し、期待値と異なる場合はエラーメッセージを出して処理を中止すること。

* 24画像がすべて生成されていること
* 各画像に、ファイル名が示す通りの CICP、CLLI、ICC Profile が存在する、または存在しないこと
* CICP の値が 9-16-0-1 であること
* CLLI の MaxCLL と MaxFALL がともに 203 nits であること
* 埋め込んだ ICC Profile に CICP = 9-16-0-1 が含まれていること
* メタデータの追加または削除だけを行う処理では、処理前後で RGB 値または YCbCr 値が変化していないこと
* AVIF、HEIC、JPEG XL はデコード後の画像を使用して画素値を比較すること
* `hdr_judge_check.html` から24画像すべてにリンクできること
