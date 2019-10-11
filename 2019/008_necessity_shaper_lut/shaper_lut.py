#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
shaper lut の必要性を主張
"""

# 外部ライブラリのインポート
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plot_utility as pu
from colour import RGB_to_RGB
from colour.models import BT2020_COLOURSPACE, BT709_COLOURSPACE
from colour.models import eotf_ST2084


# 自作ライブラリのインポート
import TyImageIO as tyio
import lut


SRC_IMG = "./src_img/Gamma 2.4_ITU-R BT.2020_D65_1920x1080_rev03_type1.exr"
DST_FILE_NAME_PREFIX = "./dst_img/"
COLOUR_LIST = ["#FF4800", "#03AF7A", "#005AFF"]

def tiff_file_write(filename, img):
    """
    OpenCV の BGR 配列が怖いので並べ替えるwrapperを用意。
    """
    cv2.imwrite(filename, img[:, :, ::-1])


def tiff_file_read(filename):
    """
    OpenCV の BGR 配列が怖いので並べ替えるwrapperを用意。
    """
    img = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)

    if img is not None:
        return img[:, :, ::-1] / 0xFFFF
    else:
        return img


def exr_file_read(fname):
    reader = tyio.TyReader(fname)
    return reader.read()


def get_log2_x_scale(
        sample_num=32, ref_val=1.0, min_exposure=-6.5, max_exposure=6.5):
    """
    Log2スケールのx軸データを作る。

    Examples
    --------
    >>> get_log2_x_scale(sample_num=10, min_exposure=-4.0, max_exposure=4.0)
    array([[  0.0625       0.11573434   0.214311     0.39685026   0.73486725
              1.36079      2.5198421    4.66611616   8.64047791  16.        ]])
    """
    x_min = np.log2(ref_val * (2 ** min_exposure))
    x_max = np.log2(ref_val * (2 ** max_exposure))
    x = np.linspace(x_min, x_max, sample_num)

    return 2.0 ** x


def shaper_func_linear_to_log2(
        x, mid_gray=0.18, min_exposure=-6.5, max_exposure=6.5):
    """
    ACESutil.Lin_to_Log2_param.ctl を参考に作成。
    https://github.com/ampas/aces-dev/blob/master/transforms/ctl/utilities/ACESutil.Lin_to_Log2_param.ctl

    Parameters
    ----------
    x : array_like
        linear data.
    mid_gray : float
        18% gray value on linear scale.
    min_exposure : float
        minimum value on log scale.
    max_exposure : float
        maximum value on log scale.

    Returns
    -------
    array_like
        log2 value that is transformed from linear x value.

    Examples
    --------
    >>> shaper_func_linear_to_log2(
    ...     x=0.18, mid_gray=0.18, min_exposure=-6.5, max_exposure=6.5)
    0.5
    >>> shaper_func_linear_to_log2(
    ...     x=np.array([0.00198873782209, 16.2917402385])
    ...     mid_gray=0.18, min_exposure=-6.5, max_exposure=6.5)
    array([  1.58232402e-13   1.00000000e+00])
    """
    # log2空間への変換。mid_gray が 0.0 となるように補正
    y = np.log2(x / mid_gray)

    # min, max の範囲で正規化。
    y_normalized = (y - min_exposure) / (max_exposure - min_exposure)

    y_normalized[y_normalized < 0] = 0

    return y_normalized


def shaper_func_log2_to_linear(
        x, mid_gray=0.18, min_exposure=-6.5, max_exposure=6.5):
    """
    ACESutil.Log2_to_Lin_param.ctl を参考に作成。
    https://github.com/ampas/aces-dev/blob/master/transforms/ctl/utilities/ACESutil.Log2_to_Lin_param.ctl

    Log2空間の補足は shaper_func_linear_to_log2() の説明を参照

    Examples
    --------
    >>> x = np.array([0.0, 1.0])
    >>> shaper_func_log2_to_linear(
    ...     x, mid_gray=0.18, min_exposure=-6.5, max_exposure=6.5)
    array([0.00198873782209, 16.2917402385])
    """
    x_re_scale = x * (max_exposure - min_exposure) + min_exposure
    y = (2.0 ** x_re_scale) * mid_gray
    # plt.plot(x, y)
    # plt.show()

    return y


def make_simple_bt2020_to_bt709_3dlut(grid_num=65):
    x_3d = lut.make_3dlut_grid(grid_num)
    temp_3d = RGB_to_RGB(x_3d, BT2020_COLOURSPACE, BT709_COLOURSPACE)
    temp_3d = np.clip(temp_3d, 0.0, 1.0)
    y_3d = temp_3d ** (1/2.4)
    lut_fname = "./luts/linear_bt2020_to_gamma2.4_bt709.spi3d"
    lut.save_3dlut(
        lut=y_3d, grid_num=grid_num, filename=lut_fname)


def make_shaper_plus_bt2020_to_bt709_3dlut(grid_num=65, sample_num_1d=1024):
    mid_gray = 0.18
    min_exposure = -20.0
    max_exposure = 10

    # make shaper 1dlut
    x_shaper = np.linspace(0, 1, sample_num_1d)
    y_shaper = shaper_func_log2_to_linear(
        x=x_shaper, mid_gray=mid_gray,
        min_exposure=min_exposure, max_exposure=max_exposure)
    lut_1d_fname = "./luts/linear_bt2020_to_gamma2.4_bt709_shaper.spi1d"
    lut.save_1dlut(lut=y_shaper, filename=lut_1d_fname)

    # make 3dlut with shaper
    x_3d = lut.make_3dlut_grid(grid_num)
    temp_3d = shaper_func_log2_to_linear(
        x=x_3d, mid_gray=mid_gray,
        min_exposure=min_exposure, max_exposure=max_exposure)
    temp_3d = RGB_to_RGB(temp_3d, BT2020_COLOURSPACE, BT709_COLOURSPACE)
    temp_3d = np.clip(temp_3d, 0.0, 1.0)
    y_3d = temp_3d ** (1/2.4)
    lut_fname = "./luts/linear_bt2020_to_gamma2.4_bt709_with_shaper.spi3d"
    lut.save_3dlut(
        lut=y_3d, grid_num=grid_num, filename=lut_fname)


def convert_from_bt2020_to_bt709_using_formula():
    """
    数式を使用した色域変換を実行する。
    入力：Linear
    出力：2.4
    """
    src_img = exr_file_read(SRC_IMG)
    temp_img = RGB_to_RGB(src_img, BT2020_COLOURSPACE, BT709_COLOURSPACE)
    temp_img = np.clip(temp_img, 0.0, 1.0)
    dst_img = np.uint16(np.round((temp_img ** (1/2.4)) * 0xFFFF))
    dst_img_fname = DST_FILE_NAME_PREFIX + "formula.tiff"
    tiff_file_write(filename=dst_img_fname, img=dst_img)


def plot_shaper_w_wo_data():
    """
    shaperありなしのデータをプロットする。
    """
    h_st = 57
    v_st = 262
    x_len = 1024

    ref = tiff_file_read("./dst_img/formula.tiff")
    w_shaper = tiff_file_read("./dst_img/with_shaper.tiff")
    wo_shaper = tiff_file_read("./dst_img/without_shaper.tiff")

    x = np.arange(x_len)
    ref_y = ref[v_st, h_st:h_st+x_len, 1]
    w_shaper_y = w_shaper[v_st, h_st:h_st+x_len, 1]
    wo_shaper_y = wo_shaper[v_st, h_st:h_st+x_len, 1]

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Title",
        graph_title_size=None,
        xlabel="Horizontal Index",
        ylabel="Output Code Value (10bit)",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=[x * 128 for x in range(9)],
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(x, ref_y, color=COLOUR_LIST[0], label='ref')
    ax1.plot(
        x, w_shaper_y, '--', color=COLOUR_LIST[1], label='with_shaper')
    ax1.plot(
        x, wo_shaper_y, '--', color=COLOUR_LIST[2], label='tetrahedral interpolation')
    plt.legend(loc='upper left')
    plt.show()


def main_func():
    # 単純な 3DLUT 作成
    grid_num = 65
    shaper_lut_sample_num = 1024
    make_simple_bt2020_to_bt709_3dlut(grid_num=grid_num)
    make_shaper_plus_bt2020_to_bt709_3dlut(
        grid_num=grid_num, sample_num_1d=shaper_lut_sample_num)
    convert_from_bt2020_to_bt709_using_formula()
    plot_shaper_w_wo_data()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
