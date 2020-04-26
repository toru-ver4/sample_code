# 結論

3DLUT の作成に成功した。作成した3DLUTをHDRコンテンツおよび自作のテストパターンに適用した例を図1～図3に示す。

各図には4つの画像が貼り付けてある。内訳は以下の通りである。

* 左上： オリジナルのHDRコンテンツ
* 右上： 以前にブログで作成した輝度マップ[2]
* 左下： 【今回作成】Y成分に準じた輝度マップ
* 右下： 【今回作成】RGBの各チャネルの Code Value 最大値に相当する輝度に準じた輝度マップ

| 名称 | 図 |
|:---|:---:|
|図1|  [f:id:takuver4:20200426175454j:plain]   |
|図2|  [f:id:takuver4:20200426175512j:plain]   |
|図3|  [f:id:takuver4:20200426175530p:plain]   |

作成した 3DLUT 一式は以下に置いてある。もしも興味のある方がいれば自由に使って頂いて構わない（トラブル発生しても責任は取れません）。

[3DLUT一式のダウンロード](https://drive.google.com/open?id=1_pHdP0Nx9wLPf7wUjWkUm_2TmK52eaH-)

# 作り方

## 概要

大したことはしてないので簡潔に説明する。以下の図4に示す通りである。

<figure class="figure-image figure-image-fotolife" title="図4. 作成した輝度マップの処理内容">[f:id:takuver4:20200426180832p:plain]<figcaption>図4. 作成した輝度マップの処理内容</figcaption></figure>

図4から分かるように輝度マップは3つの領域に分けて作成した。

* ①SDRの輝度領域(0～100nits)
* ②HDRの指定した輝度領域(100～1000nits)
* ③HDRの指定外の超高輝度領域(1000～10000nits)

まず、Linear に戻した後で Y成分を計算した。係数は RGB to XYZ 変換の Y の係数を利用した。

その後、①のSDRはグレースケールとして処理し、②のHDRは Turbo を使って色を塗り、③のHDRは Majentaでベタ塗りした。なお、図中の Turbo とは Google AI Blog  にて紹介されていた Rainbow Colormap のことである[3]。

コードは `make_luminance_map_3dlut.py` の `make_3dlut_for_luminance_map` のところである。最初は30行くらいで書けると思っていたのだが、全然そんなことはなかった。色々と調整しているうちに難解なコードになってしまった…。

## 各チャネルの最大 Code Value 値に基づいたマップ

HDRコンテンツを解析する上では、Y成分だけでなく R, G, B の各チャネルの最大値について調べることも重要である（理由は…申し訳ありませんが省略させて下さい）。そこで R, G, B の各チャネルの最大値の Code Value から算出した輝度のマップも作成した。

図1～図3 の右下の結果がそれに該当する。作成の際は図4の「RGB to Y 変換」で Y を算出するのではなく、R, G, B チャネルのうちの最大値 の Code Value に相当する ST2084 の輝度を算出するようにした。

# 考察というか感想

以下のように動画に対して複数の3DLUTを適用して同時再生すると楽しい！

# 参考資料

[1] HDTVTest, "[HDR] Star Wars: The Rise of Skywalker 4K Blu-ray HDR Analysis", https://www.youtube.com/watch?v=Hssiak8c4Js&feature=youtu.be

[2] toruのブログ, "Turbo を利用した HDR10 画像の輝度マップ作成", https://trev16.hatenablog.com/entry/2019/08/25/131001

[3] Google AI Blog, "Turbo, An Improved Rainbow Colormap for Visualization", https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html