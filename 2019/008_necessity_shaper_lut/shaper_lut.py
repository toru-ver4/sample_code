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
from colour import RGB_to_RGB
from colour.models import BT2020_COLOURSPACE, BT709_COLOURSPACE
from colour.models import eotf_ST2084


# 自作ライブラリのインポート
import TyImageIO as tyio
import lut


SRC_IMG = "./src_img/Gamma 2.4_ITU-R BT.2020_D65_1920x1080_rev03_type1.exr"


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
    min_exposure = -5.0
    max_exposure = 5.0
    x_shaper = np.linspace(0, 1, sample_num_1d)
    y_shaper = shaper_func_linear_to_log2(
        x=x_shaper, mid_gray=mid_gray,
        min_exposure=min_exposure, max_exposure=max_exposure)


def convert_from_bt2020_to_bt709_using_formula():
    """
    数式を使用した色域変換を実行する。
    入力：Linear
    出力：Linear
    """
    pass


def main_func():
    # 単純な 3DLUT 作成
    make_simple_bt2020_to_bt709_3dlut()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
    print(eotf_ST2084(1.0/1023) / 100.0)
    a = shaper_func_log2_to_linear(0.0, mid_gray=0.18, min_exposure=-20, max_exposure=4)
    print(a)