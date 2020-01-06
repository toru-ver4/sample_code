# xyY 色空間での Color Volume をプロットする

## はじめに

この記事は「Report ITU-R BT.2407 の Annex2 を実装」シリーズの1回目です。完走するか微妙ですが少しずつ記事を書いていきます。

## 背景

少し前にブログで [YouTube の HDR to SDR 変換](https://trev16.hatenablog.com/entry/2019/11/19/220840) の解析を行った。この際に Gamut Mapping に関する知識と経験が少なすぎると痛感したため、勉強のためにアルゴリズムの公開されている方式を1つ実装してみることにした。実装対象は BT.2407 の Annex2（NHKの提案する BT.2020 色域 to BT.709 色域変換）である。

BT.2407 の Annex2(以後、BT.2407と略す) の実装には、CIELAB色空間において BT.2020色域および BT.709色域の Gamut Bounary(※1) の算出が必要である。Gamut Boundary の算出方法は AMD の [FreeSync2 HDR の解説](https://gpuopen.com/using-amd-freesync-2-hdr-gamut-mapping/) ページ[2]でも言及されているが、今回は勉強も兼ねて筆者の独自方式で算出することにした。

しかし、いきなり CIELAB色空間の Gamut Boundary を算出するのは少々敷居が高かったので、まずは xyY色空間の BT.709色域の Gamut Boundary を算出することにした。

## 目的

* xyY色空間で BT.709色域の Gamut Boundary を算出する。

## 結論

筆者独自方式で xyY色空間での BT.709色域の Gamut Bounadry の算出に成功した。結果を図1に示す。

## 算出方法

### 方針

以下の動画のように、xyY 色空間を xy平面の Y方向に対する積み重ねとして表現することで Gamut Boundary を算出することにした。したがって、解くべき問題は **xyY色空間における 任意の Y に対する xy平面の Gamut Boundary の算出** となった。

### 理論



---

**ここに動画を貼る**

---





## 参考資料

[1] Report ITU-R BT.2407, "Colour gamut conversion from Recommendation ITU-R BT.2020 to
Recommendation ITU-R BT.709", https://www.itu.int/pub/R-REP-BT.2407

[2]AMD, "Using AMD FreeSync 2 HDR: Gamut Mapping", 2019, https://gpuopen.com/using-amd-freesync-2-hdr-gamut-mapping/