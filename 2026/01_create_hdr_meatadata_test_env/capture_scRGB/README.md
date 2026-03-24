# capture_scRGB

Windows の `Windows.Graphics.Capture` を使って、指定ディスプレイを `R16G16B16A16_FLOAT` で 1 フレーム取得し、`JPEG XR (.jxr)` として保存する小さな検証ツールです。

HDR に対応したアプリケーションが行う Tone / Gamut Mapping 処理の解析を目的としています。

## 概要

- 入力: 指定したディスプレイ番号
- 処理: モニタ全体を `Windows.Graphics.Capture` でキャプチャし、GPU テクスチャを CPU 読み取り用にコピー
- 出力: `64bpp RGBA Half` の `JPEG XR (.jxr)` ファイル

## ビルド方法

前提:

- Windows 11
- HDR が有効な表示環境
- Visual Studio Build Tools 2022
- CMake

Visual Studio Build Tools が未導入なら、以下で入れられます。

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools `
  --accept-package-agreements --accept-source-agreements `
  --silent `
  --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

その後、`x64 Native Tools Command Prompt for VS 2022` でビルドします。

```powershell
cd capture_scRGB
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

生成物は `build/my_capture_app.exe` です。

## 使い方

```powershell
.\build\my_capture_app.exe <display-number> <output.jxr>
```

例:

```powershell
.\build\my_capture_app.exe 1 output.jxr
```

- `display-number`: 1 始まりのディスプレイ番号
- `output.jxr`: 保存先ファイル名

指定番号が範囲外なら、利用可能なディスプレイ一覧が表示されます。

## テスト

制作者が事前に OS のスクリーンショット機能で取得した `img/dst_windows_official_screenshot.jxr` と `my_capture_app.exe` で取得した `img/dst_capture_scRGB_screenshot.jxr` を比較する簡易テストがあります。

```powershell
python .\test_capture_scRGB.py
```

Python 側では `numpy` と `imagecodecs` が必要です。
