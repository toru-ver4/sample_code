#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ST2084-BT.2020-D65 のデータを 2.4-BT.709-D65 に変換する3DLUTを作る。
ソースが ST2084 なので display reffered な変換とする（system gamma を考慮しない）。
"""

# 外部ライブラリのインポート
import os
import numpy as np
from colour.models import eotf_ST2084
from colour import RGB_to_RGB
from colour.models import BT2020_COLOURSPACE, BT709_COLOURSPACE

# 自作ライブラリのインポート
import lut

NOMINAL_WHITE_LUMINANCE = 100


def save_3dlut_csv_with_10bit_depth(grid_num, rgb_in, rgb_out):
    """
    ブログ添付用に10bit精度のLUTのcsvファイルを作成する。

    Parameters
    ----------
    grid_num : integer
        A number of grid points.
    rgb_in : ndarray
        Grid point data.
    rgb_out : ndarray
        A 3dlut data.
    """
    filename = "./3dlut_for_blog.csv"
    # 10bit整数型に変換
    rgb_i = np.uint16(np.round(rgb_in * 1023))
    rgb_o = np.uint16(np.round(rgb_out * 1023))
    with open(filename, "w") as f:
        # header
        buf = "{}, {}, {}, {}, {}, {}, {}\n".format(
            "index", "R_in", "G_in", "B_in", "R_out", "G_out", "B_out")
        f.write(buf)

        # body
        for index in range(grid_num ** 3):
            buf = "{}, {}, {}, {}, {}, {}, {}\n".format(
                index,
                rgb_i[0, index, 0], rgb_i[0, index, 1], rgb_i[0, index, 2],
                rgb_o[index, 0], rgb_o[index, 1], rgb_o[index, 2])
            f.write(buf)


def make_3dlut_grid(grid_num=33):
    """
    3DLUTの格子点データを作成

    Parameters
    ----------
    grid_num : integer
        A number of grid points.

    Returns
    -------
    ndarray
        An Array of the grid points.
        The shape is (1, grid_num ** 3, 3).

    Examples
    --------
    >>> make_3dlut_grid(grid_num=3)
    array([[[0. , 0. , 0. ],
            [0.5, 0. , 0. ],
            [1. , 0. , 0. ],
            [0. , 0.5, 0. ],
            [0.5, 0.5, 0. ],
            [1. , 0.5, 0. ],
            [0. , 1. , 0. ],
            [0.5, 1. , 0. ],
            [1. , 1. , 0. ],
            [0. , 0. , 0.5],
            [0.5, 0. , 0.5],
            [1. , 0. , 0.5],
            [0. , 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [1. , 0.5, 0.5],
            [0. , 1. , 0.5],
            [0.5, 1. , 0.5],
            [1. , 1. , 0.5],
            [0. , 0. , 1. ],
            [0.5, 0. , 1. ],
            [1. , 0. , 1. ],
            [0. , 0.5, 1. ],
            [0.5, 0.5, 1. ],
            [1. , 0.5, 1. ],
            [0. , 1. , 1. ],
            [0.5, 1. , 1. ],
            [1. , 1. , 1. ]]])
    """
    # np.meshgrid を使って 3次元の格子点座標を生成
    x = np.linspace(0, 1, grid_num)
    rgb_mesh_array = np.meshgrid(x, x, x)

    # 後の処理を行いやすくするため shape を変える
    rgb_mesh_array = [x.reshape(1, grid_num ** 3, 1) for x in rgb_mesh_array]

    # 格子点のデータ増加が R, G, B の順となるように配列を並べ替えてから
    # np.dstack を使って結合する
    rgb_grid = np.dstack(
        (rgb_mesh_array[2], rgb_mesh_array[0], rgb_mesh_array[1]))

    return rgb_grid


def main(grid_num=65):
    # R, G, B の grid point データを準備
    x = make_3dlut_grid(grid_num=grid_num)

    # 3DLUT を適用する対象は ST2084 の OETF が掛かっているため、
    # ST2084 の EOTF を掛けて linear に戻す
    linear_luminance = eotf_ST2084(x)

    # 単位が輝度(0～10000 nits)になっているので
    # 一般的に使われる 1.0 が 100 nits のスケールに変換
    linear = linear_luminance / NOMINAL_WHITE_LUMINANCE

    # 色域を BT.2020 --> BT.709 に変換
    linear_bt709 = RGB_to_RGB(RGB=linear,
                              input_colourspace=BT2020_COLOURSPACE,
                              output_colourspace=BT709_COLOURSPACE)

    # BT.709 の範囲外の値(linear_bt709 < 0.0 と linear_bt709 > 1.0 の領域)をクリップ
    linear_bt709 = np.clip(linear_bt709, 0.0, 1.0)

    # BT.709 のガンマ(OETF)をかける
    non_linear_bt709 = linear_bt709 ** (1 / 2.4)

    # 自作の LUTライブラリのクソ仕様のため shape を変換する
    lut_for_save = non_linear_bt709.reshape((grid_num ** 3, 3))

    # 自作の LUTライブラリを使って .cube の形式で保存
    lut_fname = "./st2084_bt2020_to_gamma2.4_bt709.cube"
    lut.save_3dlut(
        lut=lut_for_save, grid_num=grid_num, filename=lut_fname)

    save_3dlut_csv_with_10bit_depth(
        grid_num=grid_num, rgb_in=x, rgb_out=lut_for_save)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
    # x = make_3dlut_grid(16)
    # y = np.uint32(np.round(x * 1023))
    # print(y[:, :17, 0])
