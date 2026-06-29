# Exif Tool を使った画像データへの ICC Profile の埋め込み

## 仕様

以下のインターフェースを持つ Python 関数を `ty_lib\test_pattern_generator2.py` に追加すること。

```python
def add_icc_profile_using_exiftool(
        input_img_fname: str, output_img_fname: str,
        icc_profile_fname: str) -> None:
    pass
```

正常終了時の戻り値は `None` とする。

## 実装の方針

**個人開発** の環境であることを念頭に、異常系や非機能要件を作り込みすぎないこと。
正常系が正しく動くことを優先し、異常を検出した場合は、分かりやすいエラーメッセージを伴う例外を送出して処理を中止すること。

## 対応形式

対応する画像フォーマットは以下に限定する。

* AVIF (.avif)
* PNG (.png)
* JPEG XL (.jxl)
* HEIF (.heif)
* HEIC (.heic)

入力画像と出力画像の両方を拡張子で判定し、次の条件を満たさない場合は処理を中止すること。

* 拡張子が対応形式に含まれること
* 入力と出力の拡張子が同じであること
* 拡張子の大文字・小文字は区別しないこと

画像データの内容による形式判定は不要とする。

## 入出力ファイル

* 入力画像、出力画像、ICC Profile のファイル名が指定されていること
* 入力画像と ICC Profile が存在すること
* 入力画像と出力画像は別ファイルであること
* 上記を満たさない場合は処理を中止すること

## ICC Profile の埋め込み

ICC Profile の埋め込みには ExifTool (`exiftool`) を使用すること。以下は PNG の実行例である。

```shell
exiftool -o output.png '-ICC_Profile<=profile.icc' input.png
```

* 既存の ICC Profile がある場合は、新しい ICC Profile で置き換えて元の Profile は破棄すること
* ICC Profile の追加または置換によって、画像データの RGB 値または YCbCr 値を変更しないこと
* ExifTool の終了コードが 0 なら成功とし、それ以外は処理失敗とすること
* 埋め込み結果の追加検証は不要とすること

## エラー処理

独自例外は定義せず、`ValueError`、`FileNotFoundError`、`RuntimeError` などの標準例外を使用すること。
`exiftool` が見つからない場合や、ExifTool が異常終了した場合も、エラーメッセージを伴う例外を送出して処理を中止すること。

## 失敗時の状態維持

異常を検出した場合は ExifTool を実行しないこと。
ExifTool の実行に失敗した場合も、入力画像および既存の出力ファイルを変更しないこと。
必要に応じて一時ファイルへ出力し、正常終了後に出力ファイルへ置き換えること。

## テスト

テストコードは `ty_lib/test/test_test_pattern_generator2.py` に書くこと

### 共通

ICC プロファイルの付与前、付与後で RGB値 or YCbCr 値が変わらないことを確認する。

テストソースとなる画像は `2019/012_colour_v0.3.14_check/img/Gamma 2.4_ITU-R BT.709_D65_1920x1080_rev07_type1.png` を使用すること。
これを事前に各種画像フォーマットに lossless 形式 (Full Range とすること) で変換した後の画像をテスト画像として使用すること。
なお、lossless形式で画像変換後に `2019/012_colour_v0.3.14_check/img/Gamma 2.4_ITU-R BT.709_D65_1920x1080_rev07_type1.png` と値が異なるのは許可する。例えば `2019/012_colour_v0.3.14_check/img/Gamma 2.4_ITU-R BT.709_D65_1920x1080_rev07_type1.png` は 16-bit PNG だが、
AVIF は 10-bit で試験するので、16-bit to 10-bit 変換が働くので `2019/012_colour_v0.3.14_check/img/Gamma 2.4_ITU-R BT.709_D65_1920x1080_rev07_type1.png` を正解データとせずに、変換後の AVIF を正解データとすること

PNG ファイルの RGB値確認には `ty_lib/test_pattern_generator2.py` の `img_read` を使うこと

### AVIF

* bit深度は 10-bit とする
* 正解画像は `avifenc` を使って生成すること
  * その際に lossless として生成すること
* AVIF の RGB値の確認には `avifdec` を使って lossless で PNG 形式に変換した後に比較すること
* 生成後に `avifdec --info output.avif` のようにコマンドを実行し、エラー系のメッセージが表示されないのを確認すること

### PNG

* bit深度は 16-bit とする
* 生成後に `pngcheck output.png` のようにしてコマンドを実行し `ERROR` が表示されないのを確認すること

### JPEG XL

* bit深度は 16-bit とする（元の PNG が 16-bitなので特に指定は不要のはず）
* 正解画像は `cjxl` に `-q 100` オプションを付けて lossless で作ること
* JPEG XL の RGB値の確認には `djxl` を使って 16-bit PNG に変換して比較すること
* 生成後に `jxlinfo output.jxl` のようにコマンドを実行し、エラー系のメッセージが表示されないのを確認すること

### HEIF (.heif) / HEIC (.heic)

* bit深度は 10-bit とする
* 正解画像は `heif-enc` に `--lossless` オプションを付けて lossless で作ること
* HEIF の RGB値の確認には `heif-dec` をつかって 16-bit PNG に変換して比較すること
* 生成後に `heif-info output.heic` のようにコマンドを実行し、エラー系のメッセージが表示されないのを確認すること
