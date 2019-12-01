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
  * 0.0～1.0 の範囲外の入力値のクリップ確認
  * 0.0～1.0 の範囲外のLUT値を通った後の出力値確認
* apply 1DLUT to image data
  * 0.0～1.0 の範囲外の入力値のクリップ確認
  * 0.0～1.0 の範囲外のLUT値を通った後の出力値確認
* apply 1DLUT の逆変換 ← 無理だった
* 3DLUT の精度確認

"""

# import standard libraries
import os
import linecache

# import third-party libraries
import numpy as np
from colour import read_LUT, write_LUT
from colour import LUT1D, LUT3D
from colour import write_image, read_image
import matplotlib.pyplot as plt

# import my libraries
import plot_utility as pu

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

    # ライブラリを使って1.0を超える値を読み込み
    """cube は 1x3 に拡張されてるので 0列目だけ抽出して比較"""
    lut1d_cube = read_LUT(fname_cube).table[..., 0]

    lut1d_spi1d = read_LUT(fname_spi1d).table
    np.testing.assert_allclose(lut1d_cube, lut, rtol=1e-07)
    np.testing.assert_allclose(lut1d_spi1d, lut, rtol=1e-07)


def over_range_input_for_3dlut():
    """
    3DLUT への入力が 0.0～1.0 の範囲外だった場合のクリッピングの挙動確認。
    """
    # linear な特性の 3DLUT を作成
    lut = LUT3D.linear_table(size=5)
    lut3d = LUT3D(lut, name="over_range_input_test")

    # ソース画像＆理想の画像用意
    src_img = np.ones((256, 256, 3))
    mono_line = np.linspace(0, 1, 256) * 4 - 2
    color_line = np.dstack((mono_line, mono_line, np.zeros_like(mono_line)))
    src_img = src_img * color_line
    ideal_img = np.clip(src_img, 0.0, 1.0)

    # 3DLUT適用
    dst_img = lut3d.apply(src_img)

    # 確認
    np.testing.assert_allclose(dst_img, ideal_img, rtol=1e-07)


def over_range_output_for_3dlut():
    """
    3DLUT の出力が 0.0～1.0 の範囲外だった場合に出力値が
    正しく 0.0～1.0 の範囲外の値を持っていることを確認。
    """
    # over range な 3DLUT を作成
    lut = LUT3D.linear_table(size=5) * 4.0 - 2.0
    lut3d = LUT3D(lut, name="over_range_output_test")

    # ソース画像＆理想の画像用意
    src_img = np.ones((256, 256, 3))
    mono_line = np.linspace(0, 1, 256)
    color_line = np.dstack((mono_line, mono_line, np.zeros_like(mono_line)))
    src_img = src_img * color_line
    ideal_img = src_img * 4.0 - 2.0

    # 3DLUT適用
    dst_img = lut3d.apply(src_img)

    # 確認
    np.testing.assert_allclose(dst_img, ideal_img, rtol=1e-07)


def over_range_input_for_1dlut():
    """
    1DLUT への入力が 0.0～1.0 の範囲外だった場合のクリッピングの挙動確認。
    """
    # linear な特性の 3DLUT を作成
    lut = LUT1D.linear_table(size=16)
    lut1d = LUT1D(lut, name="over_range_input_test")

    # ソース画像＆理想の画像用意
    src_img = np.ones((256, 256, 3))
    mono_line = np.linspace(0, 1, 256) * 4 - 2
    color_line = np.dstack((mono_line, mono_line, np.zeros_like(mono_line)))
    src_img = src_img * color_line
    ideal_img = np.clip(src_img, 0.0, 1.0)

    # 3DLUT適用
    dst_img = lut1d.apply(src_img)

    # 確認
    np.testing.assert_allclose(dst_img, ideal_img, rtol=1e-07)


def over_range_output_for_1dlut():
    """
    1DLUT の出力が 0.0～1.0 の範囲外だった場合に出力値が
    正しく 0.0～1.0 の範囲外の値を持っていることを確認。
    """
    # over range な 1DLUT を作成
    lut = LUT1D.linear_table(size=16) * 4.0 - 2.0
    lut1d = LUT1D(lut, name="over_range_output_test")

    # ソース画像＆理想の画像用意
    src_img = np.ones((256, 256, 3))
    mono_line = np.linspace(0, 1, 256)
    color_line = np.dstack((mono_line, mono_line, np.zeros_like(mono_line)))
    src_img = src_img * color_line
    ideal_img = src_img * 4.0 - 2.0

    # 1DLUT適用
    dst_img = lut1d.apply(src_img)

    # 確認
    np.testing.assert_allclose(dst_img, ideal_img, rtol=1e-07)


def plot_sample(filename, sample=256):
    img = read_image(filename)
    caption, _ = os.path.splitext(os.path.basename(filename))
    data = img.flatten()[:sample]
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(20, 8),
        graph_title=caption,
        graph_title_size=None,
        xlabel="X Axis Label", ylabel="Y Axis Label",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3)
    ax1.plot(data, '-o', label=caption)
    plt.legend(loc='upper left')
    plt.savefig("./blog_img/" + caption + ".png",
                bbox_inches='tight', pad_inches=0.1)
    plt.show()


def check_3dlut_using_ramdom_data():
    """
    入力値、LUT値の双方をランダム値とした場合に、
    Reference 環境(Resolve？)との差分を確認
    """
    width = 1920
    height = 1080
    grid_num = 65
    in_file = "./img/3dlut_input.exr"
    lut_file = "./luts/random_3dlut.cube"
    out_file_resolve = "./img/3dlut_output_resolve.exr"
    out_file_nuke = "./img/3dlut_output_nuke.exr"
    out_file_colour = "./img/3dlut_output_colour.exr"

    # 入力値作成
    np.random.seed(1)
    src_data = np.random.rand(height, width, 3).astype(np.float32)
    write_image(src_data, in_file, bit_depth='float32')

    # LUT値作成
    np.random.seed(2)
    lut = np.random.rand(grid_num, grid_num, grid_num, 3).astype(np.float32)
    lut3d = LUT3D(lut, size=grid_num, name="random_value")
    write_LUT(lut3d, lut_file)

    # LUT適用
    after_img_colour = lut3d.apply(read_image(in_file, bit_depth='float32'))
    write_image(after_img_colour, out_file_colour)

    # 差分比較
    """
    注意：以下のコードを実行する前に、Resolve と Nuke で3DLUTを適用した
          画像ファイルを準備しておくこと。
    """
    after_img_resolve = read_image(out_file_resolve, bit_depth='float32')
    after_img_nuke = read_image(out_file_nuke, bit_depth='float32')
    diff_resolve = (after_img_colour - after_img_resolve).flatten()
    diff_nuke = (after_img_colour - after_img_nuke).flatten()
    print("resolve max diff = ", np.max(np.abs(diff_resolve)))
    print("resolve min diff = ", np.min(np.abs(diff_resolve)))
    print("resolve std      = ", np.std(diff_resolve))
    print("nuke    max diff = ", np.max(np.abs(diff_nuke)))
    plot_error_histogram(diff_resolve)

    # 3DLUT適用後の生データの冒頭256点くらいをプロット
    # plot_sample(in_file)
    # plot_sample(out_file_colour)
    # plot_sample(out_file_resolve)
    # plot_sample(out_file_nuke)


def plot_error_histogram(data):
    caption = "Error Histogram"
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(16, 8),
        graph_title=caption,
        graph_title_size=None,
        xlabel="Error",
        ylabel=None,
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=[x * 0.001 - 0.006 for x in range(13)],
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3)
    ax1.hist(data, bins=128, range=(-0.006, 0.006))
    plt.savefig("./blog_img/" + caption + ".png",
                bbox_inches='tight', pad_inches=0.1)
    plt.show()


def apply_hdr10_to_turbo_3dlut():
    """
    HDR10の静止画に3DLUTを適用してsRGB の輝度マップを作る。
    """
    hdr_img = read_image("./img/test_src_for_youtube_upload_riku.tif")
    lut3d = read_LUT("./luts/PQ_BT2020_to_Turbo_sRGB.cube")
    luminance_map_img = lut3d.apply(hdr_img)
    write_image(luminance_map_img, "./blog_img/3dlut_sample_turbo.png",
                bit_depth='uint8')


def main_func():
    # 1DLUT/3DLUT ファイルの Write/Read 確認
    write_read_nagative_3dlut_value()
    write_read_nagative_1dlut_value()
    write_read_minimum_3dlut_value()
    write_read_minimum_1dlut_value()
    write_read_over_range_3dlut_value()
    write_read_over_range_1dlut_value()

    # 1DLUT/3DLUT ファイルの適用の確認
    over_range_input_for_3dlut()
    over_range_output_for_3dlut()
    # over_range_input_for_1dlut()  # <-- パスしなかったorz
    over_range_output_for_1dlut()

    # 3DLUT の精度確認
    check_3dlut_using_ramdom_data()

    # ブログに掲載のサンプルコード
    apply_hdr10_to_turbo_3dlut()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
