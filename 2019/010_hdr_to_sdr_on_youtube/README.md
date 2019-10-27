# YouTube の HDR to SDR 変換を調べて 3DLUT化する

## 背景

YouTube に HDR動画をアップロードすると、SDRデバイス向けにサーバー側で自動的に HDR to SDR 変換が行われる。この変換のアルゴリズムは公開されていないため、投稿者が SDR動画を目にするのは投稿後に限定されてしまう。投稿後にしか SDRの動画を確認できないのは、なかなか不便であるため、なんとかしたいと考えた。

## 目的

HDR動画の投稿前にSDRの動画がどうなるかをプレビューできるよう、YouTube の HDR to SDR変換を解析し 3DLUT化する。

## 懸念事項

HDR to SDR 変換の解析のためには、YouTube の SDR動画のデータを入手する必要がある。当然のことながら動画データのダウンロードは規約違反である。

[利用規約 - YouTube ](https://www.youtube.com/static?template=terms&hl=ja&gl=JP)
> お客様は、「ダウンロード」または同様のリンクが本コンテンツについて本サービス上でYouTubeにより表示されている場合を除き、いかなる本コンテンツもダウンロードしてはなりません。

本記事は「ダウンロードは NG だが画面スクショならギリギリセーブ」という独自の見解に基づいて書くことにしたが、常識的に考えるとアウト（あるいは限りなく黒に近いグレー）である。ただ、SDR動画の事前プレビューが出来ないのは深刻に不便なので解析は決行することにした。

## 結論

YouTube の HDR to SDR 変換をエミュレーションする 3DLUT を作成した。3DLUT を適用した結果と 実際の YouTube での変換の比較を表1に示す。

3DLUT で特に違和感なくエミュレーションできている。

| 3DLUT でのエミュレーション | YouTube による変換 (正解画像) |
|:--------------------------:|:--------------------------:|
|![3d1](./blog_img/3dlut_01.png) | ![you](./blog_img/refference_01.png) |
|![3d1](./blog_img/3dlut_02.png) | ![you](./blog_img/refference_02.png) |
|![3d1](./blog_img/3dlut_03.png) | ![you](./blog_img/refference_03.png) |
|![3d1](./blog_img/3dlut_04.png) | ![you](./blog_img/refference_04.png) |
<div style="text-align: center;">表1. HDR to SDR 変換を 3DLUT で代用した様子[1-3]</div>

なお、表1. のキャプチャ元の動画は以下である。ソースがHDRなので、HDR対応TVではHDRとして再生されることに注意して頂きたい（当たり前のことですが…）。

https://youtu.be/Frn_Eg6qw0w

## 原理

筆者は、YouTube では図1のような変換が行われていると推測している。これを図2に示すように 3DLUT でエミュレーションする。

![zu1](./blog_img/figure_1.png)
<div style="text-align: center;">図1. YouTube での HDR to SDR 変換処理の想像図</div>

![zu2](./blog_img/figure_2.png)
<div style="text-align: center;">図2. 3DLUTを使って図1の処理をエミュレーションする様子</div>

## 3DLUT の作成方法

非常にシンプルである。以下の4ステップを実行すれば良い。

1. $65^3 = 274,625$ 点のHDR形式のパッチ動画を作成
2. HDRパッチ動画を YouTube へアップロード → https://youtu.be/guop9cohJa8
3. SDR環境で 2. の動画を再生して SDR変換後の $274,625$ 点のパッチをキャプチャ
4. キャプチャデータから HDR⇔SDR の対応関係を調べて3DLUT化

一番大変なのは 3. だが、幸い今回のケースでは 139枚のキャプチャで済むので手作業でも1時強で終わる。現実的な時間で完了する作業量である。

## 感想

これで YouTube にアップする前に SDRの確認が出来るようになった。便利すぎて涙が出そうである、マジで。

その一方で、Tone Mapping, Gamut Mapping がどのように行われているか気になっている。Tone Mapping については少し調べて記事を書きたいと思っている。

## 参考資料というか評価用画像

[1] https://hdrihaven.com/hdri/?c=outdoor&h=derelict_overpass

[2] https://hdrihaven.com/hdri/?c=indoor&h=circus_arena

[3] https://hdrihaven.com/hdri/?c=indoor&h=veranda
