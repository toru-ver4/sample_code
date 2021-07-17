# README

メモを残さないと終わるので残す。

## 1. Gmaut Boundary LUT の作成

`create_gamut_boundary_lut_jzazbz.py` で `create_gamut_boundary()` を実行

## 2. Focal LUT の作成

`create_gamut_boundary_lut_jzazbz.py` で `create_focal_lut()` を実行

## 3. Hue Chroma Pattern の作成

`plot_gamut_boundary_with_lut.py` で `plot_bt2020_bt709_tp()`, `plot_p3d65_bt709_tp()` を実行

## その他のデバッグ

`plot_gamut_boundary_with_lut.py` の `plot_cups(luminance=10000)` で CzJz 平面のプロットができる。
