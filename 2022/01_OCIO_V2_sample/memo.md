encodings の種類は以下

* log
* scene-linear
* display-linear
* sdr-video
* hdr-video
* data

`to_scene_reference` は scene reference space (ACES2065-1) への変換
`from_display_reference` は display reference space (CIE XYZ D65) からの変換

`isdata` は mattes とか alpha とかに使う

あれだね、looks, view_transform 無しのシンプルなのも1つ準備しておいて、まずはセットアップがきちんと出来ているのを確認してから look とかを適用していくと失敗しづらそう。
