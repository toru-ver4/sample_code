# ICC Profile作成支援スクリプト改良

## 背景

前提条件として、`2026/04_ICC_Profile_for_HDR_Media/IccXML-0.9.8_spec.md` を事前に読んでおいて下さい。

また、テスト実施時や実装の参考となるリファレンスの xmlファイル、iccファイルは
`2026/04_ICC_Profile_for_HDR_Media/ref_icc` または `2026/04_ICC_Profile_for_HDR_Media/ref_xml` を
参照して下さい。

## 改良内容

* ty_lib/icc_profile_calc_param.py
* ty_lib/icc_profile_xml_control.py
* 2026/04_ICC_Profile_for_HDR_Media/create_icc_profile.py

上記のスクリプトに対して、以下の改良を行って下さい。

### 第1段階

ベースとなる xml ファイルの使用を不要とする改良をしたいです。

現状のコードでは、xmlの作成時に `tree = ET.parse(template_fname)` のようにして
ベースとなる xml ファイルを読み込んでいます。

一方で理論的にはゼロから xml を作ることも十分に可能なはずです。
ということで、xmlファイル作成時に外部の xml を参照しないように変えて下さい。

### 第2段階

考え中です。もしなにか提案があれば言って下さい。

## テストについて

`2026/04_ICC_Profile_for_HDR_Media/create_icc_profile.py` を実行すると、
`2026/04_ICC_Profile_for_HDR_Media/xml` と `2026/04_ICC_Profile_for_HDR_Media/icc` に
ファイルが出力されます。

今回は xml ファイルが `2026/04_ICC_Profile_for_HDR_Media/ref_xml` のデータと
一致することを確認して下さい。
ただし、`CreationDateTime`(日付け) やチェックサムなど、値がズレて当然のものは一致確認しなくて良いです。
