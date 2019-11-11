# YouTube の HDR to SDR 変換の特性を少し調べる

## 1. 背景

[前回の記事](https://trev16.hatenablog.com/entry/2019/10/27/123425) で YouTube が行っている HDR to SDR 変換をエミュレーションする 3DLUT の作成に成功した。これにより、HDR動画を YouTube へアップロードする前に SDR 変換のプレビューが出来るようになった。

その一方で、色々とコンテンツを弄っていると、この 3DLUT を完全なブラックボックスとして利用するのではなく、ある程度は変換の特性を理解した上で利用した方が有用だと考えた。なので 3DLUT の特性を少しだけ解析した。

## 2. おことわり

本記事は自分の持っている知識を総動員して頑張って作成したが、今回の記事に関連する分野は事前知識に乏しく、調査の方針から結果の考察に至るまで、見当違いの事を行っている可能性がある。注意して頂きたい。

## 3. 結論

**Gray** および **BT709の色域外の色** の変換特性について調査した結果、以下の事実分かった。

* Gray に対しては 400 cd/m2～1000 cd/m2 にかけてロールオフするような Tone Mapping が掛かっている
  * 図1 に 3DLUT より求めた特性および推測した特性値を示す
* BT.2020 to BT.709 の色域変換の際、色域外の色に対しては単純な 3x3 の行列変換ではなく、何らかの補正処理が行われている<要追加調査>
  * 図2 に単純な 3x3 の行列変換の場合との比較を示す

## 4. 調査方法

### 4.1. Gray の特性調査

HDR to SDR の Tone Mapping に関する解説記事は幾つもあるが、今回は ITU-R BT.2390[1] を参考に調査することにした。BT.2390 では図x に示すように **EETF** という概念で Tone Mapping(※) を表現している。

※BT.2390 の文中では **Display Mapping** と表現されている。理由は…本編とは関係ないので省略する。

![a](./blog_img/EETF_BT2390.png)
<div style="text-align: center;">図x. BT2390 での EETF を含めたブロック図</div>

図x はカメラで撮影した画を加工する形のため OOTF が入っている。が、今回のケースはソースが HDR(ST2084) であり OOTF は既に適用済みであるため、図x のようにモデル化する。

![a](./blog_img/MY_EETF.png)
<div style="text-align: center;">図x. 今回のケースでの 3DLUT を含めたブロック図</div>

図x のデータから EETF を抽出するため、図x の処理に加えて γ=2.4 の EOTF および ST2084 の OETF を適用する。こうすることで 3DLUT 内部で行っている(と推測している) ST2084 の EOTF と Gamma=1/(2.4) の OETF をキャンセルする事ができて EETF の特性が分かる（はず）。

![a](./blog_img/MY_EETF_INV.png)
<div style="text-align: center;">図x. 3DLUT 内部の余分な処理をキャンセルするためのブロック図</div>

実際に図xの計算を行った結果を図x、図xに示す。図xは横軸縦軸が ST2084 の Code Valueに、図x は Luminance になっている。図x で高輝度領域が青線が波打っているのはサンプル数の不足に起因するものであり、下に凸になっている箇所は誤差が大きい箇所だと推測している。

| ![aaa](./blog_img/eetf_codevalue.png)    |  ![bbb](./blog_img/eetf_luminance.png) |
|:-------:|:-------:|
| 図x       | 図x        |

図x を見ると、横軸の 0.0～0.65 あたりは直線で、0.65～0.75 は適当な曲線で緩やかにロールオフさせ、0.75以降は再び直線で表現すれば数式化できそうである。ということで以下の式(1)のように数式化した。

$$
f(x) = \left\{
\begin{array}{ll}
0.74x + 0.01175 & (0 \leq x \lt 0.653) \\
ベジェ曲線で繋ぐ(式は長いので省略) & (0.653 \leq x \lt 0.752) &  &  &  & (1)\\
0.50808 & (0.752 \geq x)
\end{array}
\right.
$$

急に登場した 0.50808, 0.653, 0.752 はそれぞれ、100cd/m2, 400cd/m2, 1000cd/m2 に対応する ST2084 の Code Value値である。従って推測した EETF の特性は「400cd/m2 以下の領域は 傾き 0.74 の直線の特性」かつ「400～1000cd/m2 で緩やかにロールオフ」する特性を持つと言うことができる。

この数式を先ほどの図に上からプロットしてみた。結果を図x、図x に示す。まあまあ良い感じに特性を模倣できているのが分かる。

| ![aaa](./blog_img/eetf_codevalue_with_approximation.png)    |  ![bbb](./blog_img/eetf_luminance_with_approximation.png) |
|:-------:|:-------:|
| 図x       | 図x        |

さて、ここまでの分析はグレー成分しか存在しないモノクロのテストパターンを利用した行った。それでは推測した Tone Mapping の特性を自然画に適用したらどうなるだろうか。試しに筆者のNASから発掘した HDR の静止画ファイルを用いて確認してみた。

結果を図x、図x に示す。これらの図はアニメーションPNGとなっており YouTube で変換した正解画像と 推測した Tone Mapping の数式で変換した画像が交互に表示される。

![a](./blog_img/riku_apng.png)
<div style="text-align: center;">図x. 神社の写真</div>

![a](./blog_img/umi_apng.png)
<div style="text-align: center;">図x. 海の写真</div>

図を見ると分かるように一致していない。色がついている箇所の特性は Tone Mapping のみでは模倣できないようである。

### 4.2. 色域外の色の調査

色域外の色がどのように変化しているか、簡単なパッチを作成して確認した。結果を図x, 図x に示す。

| ![aaa](./blog_img/xyY_plot_hdr_to_sdr.png)    |  ![bbb](./blog_img/xyY_plot_simple_mtx_conv.png) |
|:-------:|:-------:|
| 図x YouTube で変換した結果   | 図x Tone Mapping ＋ 3x3 の単純な Matrix で変換した結果 |

図x, 図x についてもう少し解説する。

## X. 参考資料

[1] Report ITU-R BT.2390-7, "High dynamic range television for production and international programme exchange", https://www.itu.int/pub/publications.aspx?lang=en&parent=R-REP-BT.2390-7-2019

[2] 


https://drive.google.com/uc?id=1u6OZPk1FentjrMIg2xkU7g4A4bZobyGE
https://drive.google.com/uc?id=1A6C5NG4WSBbQbI87fCElOVjG3RMg1Xlj
