# OBS の User-defined shader に関するメモ

## プラグインの URL

https://obsproject.com/forum/resources/obs-shaderfilter.1736/

## 内部空間のデバッグ

### `current space` が `GS_CS_SRGB` の場合

* Image で SDR画像を開くと 0.0-1.0 の Non-Linear に見える
* Window Capture で SDRアプリケーションをキャプチャすると Non-Linear に見える

### `current space` が `GS_CS_709_SCRGB` の場合

* Image で SDR画像を開くと 0.0-1.0 の Non-Linear に見える
* Window Capture で SDRアプリケーションをキャプチャすると Non-Linear に見える

## obs-shaderfilter のカスタムビルドを試す

```
winget install --id=Kitware.CMake
winget install --id=GnuWin32.DiffUtils
winget install --id=GnuWin32.Patch

winget install --id=Microsoft.VisualStudio.2022.BuildTools --exact --override "--add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.ATLMFC --includeRecommended --quiet --wait --norestart"

```

launch "x64 Native Tools Command Prompt for VS 2022"

```
# build OBS Studio
cd C:\home\build_tools
git clone --branch 31.0.3 --recursive https://github.com/obsproject/obs-studio.git
cd obs-studio
cmake --preset windows-x64
cmake --build --preset windows-x64

cd C:\home\build_tools\obs-studio
rmdir /s /q .git
rmdir /s /q .github
cd plugins
git clone --branch 2.4.3 https://github.com/exeldro/obs-shaderfilter.git

ここで plugins\CMakeLists.txt にpっちを当てる

cd ..
cmake --preset windows-x64
cmake --build --preset windows-x64

```
