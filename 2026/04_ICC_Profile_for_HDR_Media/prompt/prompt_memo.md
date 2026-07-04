# プロンプト作成のためのメモ

## ICC XML のマニュアル作成

### 背景

私は `temporary\2026\DaVinci_Photo_page\iccDEV` をビルドして使える `iccFromXml` を使って .xml から .icc を作ってきました。

メインとなるライブラリは以下です。

* ty_lib/icc_profile_calc_param.py
* ty_lib/icc_profile_xml_control.py

これに対して以下のコードでプロファイル作成をしてきました。

* 2024/04_MHC2_Profile/create_icc_profile.py
* 2026/04_ICC_Profile_for_HDR_Media/create_icc_profile.py

実行環境は docker container にあります。以下で実行できます。

```
docker run --rm \
  -v /mnt/c/Users/toruv/OneDrive/work/sample_code:/work/src \
  -w /work/src \
  takuver4/ty_env_v2:rev12 \
  iccFromXml input.xml output.icc
```

また、参考に .icc から .xml を作成したい場合は `iccToXml` を以下のように叩けば実行できます。

```
docker run --rm \
  -v /mnt/c/Users/toruv/OneDrive/work/sample_code:/work/src \
  -w /work/src \
  takuver4/ty_env_v2:rev12 \
  iccToXml input.icc output.xml
```

### 依頼内容

これから、以下のライブラリの改修・増築を Codex を使って進めようと考えています。

* ty_lib/icc_profile_calc_param.py
* ty_lib/icc_profile_xml_control.py

それにあたり、事前に `temporary\2026\DaVinci_Photo_page\iccDEV` の `iccFromXml` の仕様をまとめて欲しいです。
その仕様を元に Codex が `icc_profile_calc_param.py` と `icc_profile_xml_control.py` の変更作業を行います。

なお、ライブラリの改修・増築には既に存在しているタグの生成を方法を洗練させるだけでなく、
「今後に新しく追加される ICC Profile のタグ生成を行う」ことも含まれます。
そのため、現状の `icc_profile_xml_control.py` の内容に引っ張られ過ぎすに、
`iccFromXml` コマンドの仕様を余すことなく書き記すつもりでドキュメントを作成して下さい。

作成した仕様書は 2026/04_ICC_Profile_for_HDR_Media/IccXML-0.9.8_spec.md に吐き出して下さい。
