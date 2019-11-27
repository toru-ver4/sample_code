# -*- coding: utf-8 -*-
"""
Colour Science for Python の 3DLUT モジュールを使う
==================================================

以下の内容を確認する。

* write/read the 3dlut files(.cube, .spi3d)
  * 負の値、極端に小さい値、1.0を超える値の Write/Read も確認
* write/read the 1dlut files(.cube, .spi1d)
  * 負の値、極端に小さい値、1.0を超える値の Write/Read も確認
* apply 3DLUT to image data
  * 負の入力値のクリップ確認
  * 負のLUT値の出力値確認
  * 1.0を超える入力値のクリップ確認
  * 1.0を超えるLUT値の出力確認
* apply 1DLUT to image data
  * 負の入力値のクリップ確認
  * 負のLUT値の出力値確認
  * 1.0を超える入力値のクリップ確認
  * 1.0を超えるLUT値の出力確認

"""

# import standard libraries
import os
import linecache

# import third-party libraries
import numpy as np
from colour import read_LUT, write_LUT, write_image, read_image
from colour import LUT1D, LUT3D

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def write_read_nagative_3dlut_value():
    """
    3DLUTに負の値を書く。
    """
    # ライブラリを使って負の値を書き出し
    fname_cube = "./luts/nagative_value_3dlut.cube"
    fname_spi3d = "./luts/nagative_value.spi3d"
    lut = LUT3D.linear_table(size=3) * 2 - 1
    lut3d = LUT3D(lut, name="nagative_value_test")
    write_LUT(lut3d, fname_cube)
    write_LUT(lut3d, fname_spi3d)

    # テキストファイルを開いて負の値が書かれているのを確認
    # 最初の要素が負の値なので、そこのテキストを抜いて比較
    line_1st_cube = linecache.getline(fname_cube, 3)
    line_1st_spi3d = linecache.getline(fname_spi3d, 4)
    for data in line_1st_cube.rstrip().split(" "):
        np.testing.assert_approx_equal(float(data), -1.0, significant=7)
    for data in line_1st_spi3d.rstrip().split(" ")[3:]:
        np.testing.assert_approx_equal(float(data), -1.0, significant=7)

    # ライブラリを使って負の値を読み込み
    lut3d_cube = read_LUT(fname_cube).table
    lut3d_spi3d = read_LUT(fname_spi3d).table
    np.testing.assert_allclose(lut3d_cube, lut, rtol=1e-07)
    np.testing.assert_allclose(lut3d_spi3d, lut, rtol=1e-07)


def write_read_minimum_3dlut_value():
    """
    3DLUTに極小値を書く。
    """
    # ライブラリを使って負の値を書き出し
    fname_cube = "./luts/minimum_value_3dlut.cube"
    fname_spi3d = "./luts/minimum_value.spi3d"
    min_value = np.finfo(np.float32).tiny
    lut = LUT3D.linear_table(size=3)
    lut[0, 0, 0, :] = np.array([min_value, min_value, min_value])
    lut3d = LUT3D(lut, name="mininum_value_test")
    write_LUT(lut3d, fname_cube)
    write_LUT(lut3d, fname_spi3d)

    # テキストファイルを開いて負の値が書かれているのを確認
    # 最初の要素だけ書き換えたので、そこのテキストを抜いて比較
    line_1st_cube = linecache.getline(fname_cube, 3)
    line_1st_spi3d = linecache.getline(fname_spi3d, 4)
    for data in line_1st_cube.rstrip().split(" "):
        np.testing.assert_approx_equal(float(data), min_value, significant=7)
    for data in line_1st_spi3d.rstrip().split(" ")[3:]:
        np.testing.assert_approx_equal(float(data), min_value, significant=7)

    # ライブラリを使って負の値を読み込み
    lut3d_cube = read_LUT(fname_cube).table
    lut3d_spi3d = read_LUT(fname_spi3d).table
    np.testing.assert_allclose(lut3d_cube, lut, rtol=1e-07)
    np.testing.assert_allclose(lut3d_spi3d, lut, rtol=1e-07)


def write_read_over_range_3dlut_value():
    """
    3DLUTに1.0を超える値を書く。
    """
    # ライブラリを使って1.0を超える値を書き出し
    fname_cube = "./luts/over_range_value_3dlut.cube"
    fname_spi3d = "./luts/over_range_value.spi3d"
    lut = LUT3D.linear_table(size=3) * 10000
    lut3d = LUT3D(lut, name="over_range_value_test")
    write_LUT(lut3d, fname_cube)
    write_LUT(lut3d, fname_spi3d)

    # テキストファイルを開いて1.0を超える値が書かれているのを確認
    # 最後の要素がめっちゃオーバーなので、そこのテキストを抜いて比較
    line_1st_cube = linecache.getline(fname_cube, 29)
    line_1st_spi3d = linecache.getline(fname_spi3d, 30)
    for data in line_1st_cube.rstrip().split(" "):
        np.testing.assert_approx_equal(float(data), 10000, significant=7)
    for data in line_1st_spi3d.rstrip().split(" ")[3:]:
        np.testing.assert_approx_equal(float(data), 10000, significant=7)

    # ライブラリを使って1.0を超える値を読み込み
    lut3d_cube = read_LUT(fname_cube).table
    lut3d_spi3d = read_LUT(fname_spi3d).table
    np.testing.assert_allclose(lut3d_cube, lut, rtol=1e-07)
    np.testing.assert_allclose(lut3d_spi3d, lut, rtol=1e-07)


def write_read_nagative_1dlut_value():
    """
    1DLUTに負の値を書く。
    """
    # ライブラリを使って負の値を書き出し
    fname_cube = "./luts/nagative_value_1dlut.cube"
    fname_spi1d = "./luts/nagative_value.spi1d"
    lut = LUT1D.linear_table(size=16) * 2 - 1
    lut1d = LUT1D(lut, "negative_value_test")
    write_LUT(lut1d, fname_cube)
    write_LUT(lut1d, fname_spi1d)

    # テキストファイルを開いて負の値が書かれているのを確認
    # 最初の値が負なので、そこのテキストを抜いて比較
    line_1st_cube = linecache.getline(fname_cube, 3)
    line_1st_spi1d = linecache.getline(fname_spi1d, 6)
    for data in line_1st_cube.rstrip().split(" "):
        np.testing.assert_approx_equal(float(data), -1.0, significant=7)
    np.testing.assert_approx_equal(
        float(line_1st_spi1d.rstrip()), -1.0, significant=7)

    # ライブラリを使って負の値を読み込み
    """cube は 1x3 に拡張されてるので 0列目だけ抽出して比較"""
    lut1d_cube = read_LUT(fname_cube).table[..., 0]

    lut1d_spi1d = read_LUT(fname_spi1d).table
    np.testing.assert_allclose(lut1d_cube, lut, rtol=1e-07)
    np.testing.assert_allclose(lut1d_spi1d, lut, rtol=1e-07)


def write_read_minimum_1dlut_value():
    """
    1DLUTに極小値を書く。
    """
    # ライブラリを使って極小値を書き出し
    fname_cube = "./luts/minimum_value_1dlut.cube"
    fname_spi1d = "./luts/minimum_value.spi1d"
    min_value = np.finfo(np.float32).tiny
    lut = LUT1D.linear_table(size=16) * 2
    lut[0] = min_value
    lut1d = LUT1D(lut, name="mininum_value_test")
    write_LUT(lut1d, fname_cube)
    write_LUT(lut1d, fname_spi1d)

    # テキストファイルを開いて極小値が書かれているのを確認
    # 最初の要素だけ書き換えたので、そこのテキストを抜いて比較
    line_1st_cube = linecache.getline(fname_cube, 3)
    line_1st_spi1d = linecache.getline(fname_spi1d, 6)
    for data in line_1st_cube.rstrip().split(" "):
        np.testing.assert_approx_equal(float(data), min_value, significant=7)
    np.testing.assert_approx_equal(
        float(line_1st_spi1d.rstrip()), min_value, significant=7)

    # ライブラリを使って極小値を読み込み
    """cube は 1x3 に拡張されてるので 0列目だけ抽出して比較"""
    lut1d_cube = read_LUT(fname_cube).table[..., 0]

    lut1d_spi1d = read_LUT(fname_spi1d).table
    np.testing.assert_allclose(lut1d_cube, lut, rtol=1e-07)
    np.testing.assert_allclose(lut1d_spi1d, lut, rtol=1e-07)


def write_read_over_range_1dlut_value():
    """
    1DLUTに1.0を超える値を書く。
    """
    # ライブラリを使って1.0を超える値を書き出し
    fname_cube = "./luts/over_range_value_1dlut.cube"
    fname_spi1d = "./luts/over_range_value.spi1d"
    lut = LUT1D.linear_table(size=16) * 10000
    lut1d = LUT1D(lut, name="over_range_value_test")
    write_LUT(lut1d, fname_cube)
    write_LUT(lut1d, fname_spi1d)

    # テキストファイルを開いて1.0を超える値が書かれているのを確認
    # 最後の要素がめっちゃオーバーなので、そこのテキストを抜いて比較
    line_1st_cube = linecache.getline(fname_cube, 18)
    line_1st_spi1d = linecache.getline(fname_spi1d, 21)
    for data in line_1st_cube.rstrip().split(" "):
        np.testing.assert_approx_equal(float(data), 10000, significant=7)
    np.testing.assert_approx_equal(
        float(line_1st_spi1d.rstrip()), 10000, significant=7)

    # ライブラリを使って極小値を読み込み
    """cube は 1x3 に拡張されてるので 0列目だけ抽出して比較"""
    lut1d_cube = read_LUT(fname_cube).table[..., 0]

    lut1d_spi1d = read_LUT(fname_spi1d).table
    np.testing.assert_allclose(lut1d_cube, lut, rtol=1e-07)
    np.testing.assert_allclose(lut1d_spi1d, lut, rtol=1e-07)


def main_func():
    write_read_nagative_3dlut_value()
    write_read_nagative_1dlut_value()
    write_read_minimum_3dlut_value()
    write_read_minimum_1dlut_value()
    write_read_over_range_3dlut_value()
    write_read_over_range_1dlut_value()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
