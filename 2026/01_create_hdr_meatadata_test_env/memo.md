# 調査内容メモ

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

## CodeX 文言

今開いている main.cpp は Windows 11 の DWM が各アプリケーションの描画結果をscRGB 空間にコンポジットした結果を
キャプチャするコマンドラインアプリです。

これに以下の改良を加えて下さい。

* デュアルモニター環境でも動作するように、引数に「ディスプレイ番号」を入力する仕組みを追加すること
* 指定したディスプレイ番号の scRGB のバッファを jxr 形式で保存すること

もしも「その仕様は実現不可能である」といった事があれば行って下さい。

# ChatGPT 文言

■背景
スクリプトやプログラムを組んで以下のことをしたいと考えている。
1. Windows 11 の Chrome/Edge に指定した動画・静止画コンテンツを表示
2. F11 キーを押下に相当するコマンドを Chrome/Edge に送って全画面表示をする
3. 試験者が表示内容を目視確認
4. 手順 1.～3. を用意したコンテンツ分繰り返す

■調査して欲しい内容
上記の操作を半自動で行うための仕組みが存在していれば教えて欲しい。

