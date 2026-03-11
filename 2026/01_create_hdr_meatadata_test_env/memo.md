# 調査内容メモ

## ブラウザの自動操作について

- Playwright が今どきらしい
- 色々とできるがブラウザの起動場所の座標計算は別に必要
  - Python の screeninfo がメジャー？
  - 

## ChatGPT に投げる文言

以下の調査をお願いします。

■背景
Windows 11 で HDR を有効にした状態で、Chrome/Edge 上で様々なHDRコンテンツを表示し、
その表示結果を解析したいと考えている（トーンマッピングが適用されるのか、ガマットマッピングが適用されるのか、など）。

■調査して欲しい内容
Windows は [公式ドキュメント](https://learn.microsoft.com/en-us/windows/win32/direct3darticles/high-dynamic-range) を参照すると、各アプリケーションの描画結果を DWM にて scRGB 空間にコンポジットしている。

このコンポジット後の scRGB の生値の取得を自動的に行うためのプログラムが欲しい。
既にそうしたプログラムが存在しているなら紹介して欲しいし、
存在していないなら、どういった　API を叩けば実現できそうか調べて提示して欲しい（まだコーディングはしなくて良い。必要になったらこちらから指示する）

もし、コンポジット後の scRGB値の取得が極めて困難なのであれば、
コンポジット前の各アプリケーションのバッファ情報の取得でも構わない（Win+Alt+PrintScreen がこれに該当か？）

## ChatGPT に投げる文言

以下の調査をお願いします。

■背景
Windows 11 で HDR を有効にした状態で、Chrome/Edge 上で様々なHDRコンテンツを表示し、
その表示結果を解析したいと考えている（トーンマッピングが適用されるのか、ガマットマッピングが適用されるのか、など）。

Windows は [公式ドキュメント](https://learn.microsoft.com/en-us/windows/win32/direct3darticles/high-dynamic-range) を参照すると、各アプリケーションの描画結果を DWM にて scRGB 空間にコンポジットしている。

このコンポジット後（デスクトップ全体）の scRGB の生値の取得を自動的に行うためのプログラムを書きたい。

API としては以下のどちらかを使おうと考えている。

* Windows.Graphics.Capture (WGC)
* Desktop Duplication API

また、そのプログラムは以下のようなコマンドラインアプリを想定している。

```powershell
my_capture_app.exe output_filename.jxr
```

拡張子が .jxr なのは一例である。OpenEXR 方式でも構わない。

■調査して欲しい内容

以下のどちらの方が簡単にプログラムのソースコードを書けそうか調べて欲しい。
なお、私の環境には Visual Studio は入っていない。ゼロから環境構築する必要がある。

* Windows.Graphics.Capture (WGC)
* Desktop Duplication API

## ChatGPT に投げる文言

以下の調査をお願いします。

■背景
Windows 11 で HDR を有効にした状態で、Chrome/Edge 上で様々なHDRコンテンツを表示し、
その表示結果を解析したいと考えている（トーンマッピングが適用されるのか、ガマットマッピングが適用されるのか、など）。

Windows は [公式ドキュメント](https://learn.microsoft.com/en-us/windows/win32/direct3darticles/high-dynamic-range) を参照すると、各アプリケーションの描画結果を DWM にて scRGB 空間にコンポジットしている。

このコンポジット後（デスクトップ全体）の scRGB の生値の取得を自動的に行うためのプログラムを書きたい。

API としては以下を使おうと考えている。

* Desktop Duplication API

また、そのプログラムは以下のようなコマンドラインアプリを想定している。

```powershell
my_capture_app.exe output_filename.jxr
```

拡張子が .jxr なのは一例である。OpenEXR 方式でも構わない。

■調査して欲しい内容

Visual Studio も入っていない Windows 11 環境で、
DWM での コンポジット後（デスクトップ全体）の scRGB の生値の取得を自動的に行うためのプログラムを
書いて下さい。加えてビルド環境の整備の仕方も提示して下さい。

## 仕様変更文言

今開いている hdr10_test.py に対して以下の仕様変更をお願いします。

`get_display2_geometry` を使って geometry を設定していますが、
これはスクリプトの冒頭に定数として入力する形として下さい。

初期値としては left, top, width, height = [0, 0, 1920, 1080] を設定して下さい。（変数名や代入の仕方は変えてもらって構いません）

run_capture_exe で決め打ちで指定しているディスプレイ番号 "2" も、スクリプト上部に定数として宣言して下さい。
なお、意味合いとしてはキャプチャ対象のディスプレイ番号です。

## CodeX 文言

今開いている main.cpp は Windows 11 の DWM が各アプリケーションの描画結果をscRGB 空間にコンポジットした結果を
キャプチャするコマンドラインアプリです。

これに以下の改良を加えて下さい。

* デュアルモニター環境でも動作するように、引数に「ディスプレイ番号」を入力する仕組みを追加すること
* 指定したディスプレイ番号の scRGB のバッファを jxr 形式で保存すること

もしも「その仕様は実現不可能である」といった事があれば行って下さい。

### ChatGPT 文言

■背景
スクリプトやプログラムを組んで以下のことをしたいと考えている。
1. Windows 11 の Chrome/Edge に指定した動画・静止画コンテンツを表示
2. F11 キーを押下に相当するコマンドを Chrome/Edge に送って全画面表示をする
3. 試験者が表示内容を目視確認
4. 手順 1.～3. を用意したコンテンツ分繰り返す

■調査して欲しい内容
上記の操作を半自動で行うための仕組みが存在していれば教えて欲しい。

### ChatGPT 文言

■背景
スクリプトやプログラムを組んで以下のことをしたいと考えている。
* Windows 11 の Chrome/Edge で特定の WebページA(https://toru-ver4.github.io/pages_test/MDCV_CLLI_Test/index.html)を開く
  * なお Chrome/Edge はキオスクモードで起動し、全画面表示状態を維持するものとする
  * また試験はデュアルモニター環境で行い、スクリプトの制御は Display No.1（この No は Windows の System -> Display の No と一致）に、ブラウザの表示は Display No.2 に行うものとする
* WebページA には評価用画像・動画へのリンクが多数記載されている
* スクリプトは事前に、WebページAの評価用画像・動画へのリンクが付与されたテキストのリストを持っている
  * テキストの例は "./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-10000.png" である
* スクリプトは上から順にテキストのリストを読み取り、テキストと一致するWebページAのリンクを開き、WebページB を表示する
* WebページB を開いた後は 5秒ほど待った後、画面キャプチャのコマンド .\capture_scRGB\build\my_capture_app.exe を叩く
* コマンドの引数は `.\capture_scRGB\build\my_capture_app.exe 2 <output.jxr>`
  * ただし、<output.jxr> は .\capture_img\<リンクのテキストからフォルダ名と拡張子を取り除いたもの>.jxr とする
  * 例えば ".\capture_img\png_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-10000.jxr" となる
* 画面キャプチャが終わったあとは、WebページBを閉じて WebページAに戻る
* その後は「WebページAの評価用画像・動画へのリンクが付与されたテキストのリスト」のリンクに対して手順を繰り返す
* 「WebページAの評価用画像・動画へのリンクが付与されたテキストのリスト」は以下の通り

```
link_text_list = [
    ./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-100.mp4
    ./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-10000.mp4
    ./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-None.mp4
    ./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-100.mp4
    ./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-10000.mp4
    ./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-None.mp4
    ./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-100.mp4
    ./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-10000.mp4
    ./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-None.mp4
    ./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-100.mp4
    ./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.mp4
    ./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-None.mp4
    ./metadata_img/av1_mdcv-p-None_mdcv-l-None_clli-100.mp4
    ./metadata_img/av1_mdcv-p-None_mdcv-l-None_clli-10000.mp4
    ./metadata_img/av1_mdcv-p-None_mdcv-l-None_clli-None.mp4
    ./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-100.mp4
    ./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-10000.mp4
    ./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-None.mp4
    ./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-100.mp4
    ./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-10000.mp4
    ./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-None.mp4
    ./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-100.mp4
    ./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-10000.mp4
    ./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-None.mp4
    ./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-100.mp4
    ./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.mp4
    ./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-None.mp4
    ./metadata_img/hevc_mdcv-p-None_mdcv-l-None_clli-100.mp4
    ./metadata_img/hevc_mdcv-p-None_mdcv-l-None_clli-10000.mp4
    ./metadata_img/hevc_mdcv-p-None_mdcv-l-None_clli-None.mp4
    ./metadata_img/avif_mdcv-p-None_mdcv-l-None_clli-100.avif
    ./metadata_img/avif_mdcv-p-None_mdcv-l-None_clli-10000.avif
    ./metadata_img/avif_mdcv-p-None_mdcv-l-None_clli-None.avif
    ./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-100.png
    ./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-10000.png
    ./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-None.png
    ./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-100.png
    ./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-10000.png
    ./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-None.png
    ./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-100.png
    ./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-10000.png
    ./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-None.png
    ./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-100.png
    ./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.png
    ./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-None.png
    ./metadata_img/png_mdcv-p-None_mdcv-l-None_clli-100.png
    ./metadata_img/png_mdcv-p-None_mdcv-l-None_clli-10000.png
    ./metadata_img/png_mdcv-p-None_mdcv-l-None_clli-None.png
]
```

■調査して欲しい内容
上記の操作はスクリプト言語で可能か教えて欲しい。
また可能であれば、どの言語が楽か知りたい。
一方で Python での実装難易度も教えて欲しい。可能ならPythonで実装したい


## CodeX

あなたは Windows 自動化と Playwright に精通した Python エンジニアです。
以下の要件を満たす 実行可能な Python スクリプト一式 を作成してください。

■ 目的

Windows 11 環境で Microsoft Edge または Chrome を起動し、

・Display No.2 にウィンドウを配置
・起動時から「全画面相当」で表示（キオスクは使用しない）
・指定のリンクを順番に開く
・5秒待機
・外部キャプチャEXEを実行
・ページを閉じて元ページへ戻る
・すべてのリンクで繰り返す

■ 使用技術

・Python 3.11 以上
・Playwright (sync API)
・subprocess
・pathlib
・必要に応じて pywin32 または ctypes（Display No.2への移動に使用）

■ 起動仕様（重要）

キオスクは使用しない。
以下のいずれかの方法で「起動時から全画面相当」にすること：

優先順位：

Chromium 起動引数で全画面開始
--start-fullscreen
--start-maximized
--window-position
--window-size

起動後に Playwright API で viewport をモニタ2サイズに変更

必要なら Win32 API でウィンドウをモニタ2へ移動

■ モニタ仕様

・Display No.1 = スクリプト制御用
・Display No.2 = ブラウザ表示用
・Windows の「設定 → ディスプレイ」の番号と一致
・Display No.2 の解像度を取得し、そのサイズで表示すること
・DPIスケーリング環境でも動作すること

■ 処理対象URL

WebページA：
https://toru-ver4.github.io/pages_test/MDCV_CLLI_Test/index.html

■ リンクリスト

link_text_list = [
"./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-100.mp4",
"./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-10000.mp4",
"./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-None.mp4",
"./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-100.mp4",
"./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-10000.mp4",
"./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-None.mp4",
"./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-100.mp4",
"./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-10000.mp4",
"./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-None.mp4",
"./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-100.mp4",
"./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.mp4",
"./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-None.mp4",
"./metadata_img/av1_mdcv-p-None_mdcv-l-None_clli-100.mp4",
"./metadata_img/av1_mdcv-p-None_mdcv-l-None_clli-10000.mp4",
"./metadata_img/av1_mdcv-p-None_mdcv-l-None_clli-None.mp4",
"./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-100.mp4",
"./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-10000.mp4",
"./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-None.mp4",
"./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-100.mp4",
"./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-10000.mp4",
"./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-None.mp4",
"./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-100.mp4",
"./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-10000.mp4",
"./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-None.mp4",
"./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-100.mp4",
"./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.mp4",
"./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-None.mp4",
"./metadata_img/hevc_mdcv-p-None_mdcv-l-None_clli-100.mp4",
"./metadata_img/hevc_mdcv-p-None_mdcv-l-None_clli-10000.mp4",
"./metadata_img/hevc_mdcv-p-None_mdcv-l-None_clli-None.mp4",
"./metadata_img/avif_mdcv-p-None_mdcv-l-None_clli-100.avif",
"./metadata_img/avif_mdcv-p-None_mdcv-l-None_clli-10000.avif",
"./metadata_img/avif_mdcv-p-None_mdcv-l-None_clli-None.avif",
"./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-100.png",
"./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-10000.png",
"./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-None.png",
"./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-100.png",
"./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-10000.png",
"./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-None.png",
"./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-100.png",
"./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-10000.png",
"./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-None.png",
"./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-100.png",
"./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.png",
"./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-None.png",
"./metadata_img/png_mdcv-p-None_mdcv-l-None_clli-100.png",
"./metadata_img/png_mdcv-p-None_mdcv-l-None_clli-10000.png",
"./metadata_img/png_mdcv-p-None_mdcv-l-None_clli-None.png"
]

■ 処理仕様

各リンクについて：

WebページAを開く

該当リンクと一致する href を持つリンクをクリック
テキスト一致または href一致で可

WebページBが表示されたら5秒待機

以下コマンドを実行

.\capture_scRGB\build\my_capture_app.exe 2 <output_path>

<output_path> の生成規則

"./metadata_img/png_xxx.png"
→ "png_xxx.jxr"
→ ".\capture_img\png_xxx.jxr"

・フォルダ名削除
・拡張子削除
・.jxr に変更

ページBを閉じてWebページAへ戻る

■ 要件

・例外処理を適切に実装
・各ステップでログ出力
・途中失敗時でも次のリンクへ進める設計
・実行前に playwright install が必要である旨コメント記載
・main() エントリポイントを持つ
・Windows専用でよい

■ 出力形式

単一の hdr10_test.py ファイルとして完成形を出力

必要な pip インストール一覧を先頭コメントに記載

実行方法もコメントで記載

コード内に TODO を残さない

■ 実装の安定性優先

・クリックより page.goto() の方が安定する場合は直接URL遷移可
・ウィンドウハンドル取得が必要なら実装すること
・確実にDisplay No.2で全画面相当を最優先

■ 最後に

コードは「そのまま実行可能」な完成形で出力すること。
説明文は不要。コードのみ出力すること。

### プロット

plot_all_data() で以下のプロットを行うコードを書いて下さい。

step_ramp_7colors は shape = (7, 65, 3) の ndarray であり、
White, Yellow, Cyan, Green, Magenta, Red, Blue の各色の 65点の単調増加のデータが (R,G,B) の3刺激値として入っています。

これを subplot を使って 縦にグラフを 7つ並べて、各色のデータをプロットします。
各色のデータのプロットでは (R, G, B) の3刺激値をプロットします。
加えて reference value として x = np.linspace(1.0, 10000, 1024), x = np.linspace(1.0, 10000, 1024) を点線でプロットして下さい。

x軸、y軸のレンジは共に 1.0～10,000 です。log10 の対数グラフとしてプロットして下さい。
グラフタイトルは不要です。

軸の目盛り(ticks) は x軸もy軸も 1, 10, 100, 1000, 10000 のところに実線を引いて下さい。ただし色はグレーにして下さい。
そして、その間を埋める形で 補助線を 2, 4, 6, 8, 20, 40, 60, 80, 200, 400,600, 800, 2000, 4000, 6000, 8000 に引いて下さい。

凡例(legend) は各色ごとに記述して下さい。命名速は「色名 - 3刺激値を示すアルファベット1文字」です。
例えば、「White - R」 や 「Green - G」 や「Cyan - B」のように凡例を設定して下さい。
line color は3刺激値に合わせてください。例えば Cyan - B の line color は #0000FF のように Blue を指定して下さい。

グラフは現在は  show して下さい。後でコメントアウトできるようにしておいて下さい。
グラフは `output_graph_name`の名前で保存して下さい。

### ChatGPT

以下の内容を読んで調査をお願いします。

■背景
Playwright というブラウザ制御のフレームワークを使って様々なHDRコンテンツの描画を行い、
そのスクリーンショットを撮影して描画結果の解析をしようと考えています。

事前確認として「E2E の同等性検証」のような形で、
私が普段使っているブラウザの状態と、Playwright経由で起動するブラウザの状態が、色関連の描画で一致することを確認したいです。

■調査依頼
おそらくですが、以下のような内容を手動起動のブラウザと Playwright経由のブラウザの双方でダンプして、
値を比較するのが良いと考えています。

```
({
  dynamicRangeHigh: matchMedia("(dynamic-range: high)").matches,
  colorGamutRec2020: matchMedia("(color-gamut: rec2020)").matches,
})
```

ただ、残念ながら私はブラウザのパラメータに疎く、色・輝度に関連するパラメータにどのようなものがあるか分かりません。
お手数ですが、調査してダンプすべきパラメータの一覧を作成して下さい。

その際、ブラウザのコンソールで確認する方法と Playwright の Pythonスクリプトで確認する方法の2通りを提示して下さい。

以下を得た。

```
(() => {
  const mm = (q) => {
    try {
      return matchMedia(q).matches;
    } catch {
      return null;
    }
  };

  const safe = (fn) => {
    try {
      return fn();
    } catch (e) {
      return { error: String(e) };
    }
  };

  const canvas2dInfo = safe(() => {
    const canvas = document.createElement("canvas");

    const ctxDefault = canvas.getContext("2d");
    const ctxP3 = canvas.getContext("2d", { colorSpace: "display-p3" });
    const ctxFloat16 = canvas.getContext("2d", { colorType: "float16" });
    const ctxP3Float16 = canvas.getContext("2d", {
      colorSpace: "display-p3",
      colorType: "float16",
    });

    return {
      defaultContext: !!ctxDefault,
      displayP3Context: !!ctxP3,
      float16Context: !!ctxFloat16,
      displayP3Float16Context: !!ctxP3Float16,
    };
  });

  const webglInfo = safe(() => {
    const canvas = document.createElement("canvas");
    const gl =
      canvas.getContext("webgl2") ||
      canvas.getContext("webgl") ||
      canvas.getContext("experimental-webgl");

    if (!gl) {
      return { supported: false };
    }

    return {
      supported: true,
      drawingBufferColorSpace:
        "drawingBufferColorSpace" in gl ? gl.drawingBufferColorSpace : null,
      unpackColorSpace:
        "unpackColorSpace" in gl ? gl.unpackColorSpace : null,
    };
  });

  const result = {
    timestamp: new Date().toISOString(),
    url: location.href,
    title: document.title,
    userAgent: navigator.userAgent,

    mediaQueries: {
      dynamicRangeStandard: mm("(dynamic-range: standard)"),
      dynamicRangeHigh: mm("(dynamic-range: high)"),

      colorGamutSrgb: mm("(color-gamut: srgb)"),
      colorGamutP3: mm("(color-gamut: p3)"),
      colorGamutRec2020: mm("(color-gamut: rec2020)"),

      videoDynamicRangeStandard: mm("(video-dynamic-range: standard)"),
      videoDynamicRangeHigh: mm("(video-dynamic-range: high)"),

      videoColorGamutSrgb: mm("(video-color-gamut: srgb)"),
      videoColorGamutP3: mm("(video-color-gamut: p3)"),
      videoColorGamutRec2020: mm("(video-color-gamut: rec2020)"),

      forcedColorsNone: mm("(forced-colors: none)"),
      forcedColorsActive: mm("(forced-colors: active)"),

      prefersContrastNoPreference: mm("(prefers-contrast: no-preference)"),
      prefersContrastMore: mm("(prefers-contrast: more)"),

      prefersColorSchemeLight: mm("(prefers-color-scheme: light)"),
      prefersColorSchemeDark: mm("(prefers-color-scheme: dark)"),
    },

    screen: {
      width: screen.width,
      height: screen.height,
      availWidth: screen.availWidth,
      availHeight: screen.availHeight,
      colorDepth: screen.colorDepth,
      pixelDepth: screen.pixelDepth,
      devicePixelRatio: window.devicePixelRatio,
      orientationType: screen.orientation?.type ?? null,
      orientationAngle: screen.orientation?.angle ?? null,
      isExtended: "isExtended" in screen ? screen.isExtended : null,
    },

    canvas2d: canvas2dInfo,
    webgl: webglInfo,
  };

  const json = JSON.stringify(result, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "dump.json";
  document.body.appendChild(a);
  a.click();
  a.remove();

  setTimeout(() => URL.revokeObjectURL(url), 1000);

  return result;
})();
```