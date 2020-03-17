# -*- coding: utf-8 -*-
"""
CIELAB の Gamut Boundary データの補間
=====================================

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
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


def calc_bilinear_sample_data(lh, l_sample_num, h_sample_num):
    """
    CIELAB空間の特定の色域の Gamut Boundary に対して
    Bilinear Interpolation を行うためのサンプル点の抽出を行う。

    Parameters
    ----------
    lh : array_like
        L* and Hue data.
    l_sample_num : int
        Number of samples in Lihgtness direction.
    h_sample_num : int
        Number of samples in Hue direction.

    Returns
    -------
    array_like
        Four sample indeces used for Bilinear Interpolation.
        And two ratio data for interpolation.

    Examples
    --------
    >>> test_data = np.array(
    ...     [[0.0, 0.0], [100.0, 2 * np.pi],
    ...      [0.00005, 0.00001 * np.pi], [99.99995, 1.99999 * np.pi]])
    >>> indices, ratios = calc_bilinear_sample_data(
    ...     lh=test_data, l_sample_num=256, h_sample_num=256)
    >>> print(indices)
    >>> [[[  0.   0.   0.   0.]
    ...   [255. 255. 255. 255.]
    ...   [  1.   0.   1.   0.]
    ...   [255. 254. 255. 254.]]]
    >>> print(ratios)
    >>> [[[0.000000e+00, 0.000000e+00],
    ...   [1.275000e-04, 1.275000e-03],
    ...   [9.998725e-01, 9.987250e-01]]]
    """
    l_temp = lh[..., 0] / 100 * (l_sample_num - 1)
    h_temp = lh[..., 1] / (2 * np.pi) * (h_sample_num - 1)
    l_hi = np.uint16(np.ceil(l_temp))
    l_lo = np.uint16(np.floor(l_temp))
    h_hi = np.uint16(np.ceil(h_temp))
    h_lo = np.uint16(np.floor(h_temp))
    r_l = l_hi - l_temp  # ratio in Luminance direction
    r_h = h_hi - h_temp  # ratio in Hue direction

    return np.dstack((l_hi, l_lo, h_hi, h_lo)), np.dstack((r_l, r_h))


def bilinear_interpolation(lh, lut2d):
    """
    Bilinear で補間します。

    Parameters
    ----------
    lh : array_like
        L* and Hue data.
    lut2d : array_like
        2d lut data.

    Returns
    -------
    array_like
        four sample indeces used for Bilinear Interpolation.

    Examples
    --------
    >>> # 256x256 は密集すぎなのでスカスカな LUT を作成
    >>> chroma_lut = np.load("./boundary_data/Chroma_BT709_l_256_h_256.npy")
    >>> h_idx = np.arange(0, 256, 8)
    >>> l_idx = np.arange(0, 256, 64)
    >>> sparse_lut = chroma_lut[l_idx][:, h_idx]
    >>>
    >>> l_sample_num, h_sample_num = sparse_lut.shape
    >>>
    >>> # 補間の入力データ lh を作成
    >>> l_val = 40
    >>> target_h_num = 256
    >>> ll = l_val * np.ones(target_h_num)
    >>> hue = np.linspace(0, 2 * np.pi, target_h_num)
    >>> lh = np.dstack((ll, hue))
    >>>
    >>> # 補間実行
    >>> chroma_interpolation = bilinear_interpolation(lh, sparse_lut)
    """
    l_sample_num = lut2d.shape[0]
    h_sample_num = lut2d.shape[1]
    indices, ratios = calc_bilinear_sample_data(
        lh, l_sample_num, h_sample_num)
    l_hi_idx = indices[..., 0]
    l_lo_idx = indices[..., 1]
    h_hi_idx = indices[..., 2]
    h_lo_idx = indices[..., 3]
    l_ratio = ratios[..., 0]
    h_ratio = ratios[..., 1]

    # interpolation in Hue direction
    temp_hi = lut2d[l_hi_idx, h_hi_idx] * (1 - h_ratio)\
        + lut2d[l_hi_idx, h_lo_idx] * h_ratio
    temp_lo = lut2d[l_lo_idx, h_hi_idx] * (1 - h_ratio)\
        + lut2d[l_lo_idx, h_lo_idx] * h_ratio

    # interpolation in Luminance direction
    result = temp_hi * (1 - l_ratio) + temp_lo * l_ratio

    return result[0]


def check_interpolation():
    """"""
    # 256x256 は密集すぎなのでスカスカな LUT を作成
    chroma_lut = np.load("./boundary_data/Chroma_BT709_l_256_h_256.npy")
    h_idx = np.arange(0, 256, 8)
    l_idx = np.arange(0, 256, 64)
    sparse_lut = chroma_lut[l_idx][:, h_idx]

    l_sample_num, h_sample_num = sparse_lut.shape

    # 補間の入力データ lh を作成
    l_val = 40
    target_h_num = 256
    ll = l_val * np.ones(target_h_num)
    hue = np.linspace(0, 2 * np.pi, target_h_num)
    lh = np.dstack((ll, hue))

    # 補間実行
    chroma_interpolation = bilinear_interpolation(lh, sparse_lut)

    # 補間結果のプロット
    l_temp = l_val / 100 * (l_sample_num - 1)
    l_lo_idx = int(np.floor(l_temp))
    l_hi_idx = int(np.ceil(l_temp))

    rad = np.linspace(0, 2 * np.pi, h_sample_num)
    rad2 = np.linspace(0, 2 * np.pi, target_h_num)
    a_lo = sparse_lut[l_lo_idx] * np.cos(rad)
    b_lo = sparse_lut[l_lo_idx] * np.sin(rad)
    a_hi = sparse_lut[l_hi_idx] * np.cos(rad)
    b_hi = sparse_lut[l_hi_idx] * np.sin(rad)
    a = chroma_interpolation * np.cos(rad2)
    b = chroma_interpolation * np.sin(rad2)
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="CIELAB Plane",
        graph_title_size=None,
        xlabel="a*", ylabel="b*",
        axis_label_size=None,
        legend_size=17,
        xlim=(-100, 100),
        ylim=(-100, 100),
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(a_lo, b_lo, 'o', label="Low")
    ax1.plot(a_hi, b_hi, 'o', label="High")
    ax1.plot(a, b, 'o', label="Interpolation")
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # target = 0.9999999999999999

    # test_data = np.array(
    #     [[0.0, 0.0], [100.0, 2 * np.pi],
    #      [0.00005, 0.00001 * np.pi], [99.99995, 1.99999 * np.pi]])
    # # indices, ratios = calc_bilinear_sample_data(test_data, 256, 256)
    # # print(indices, ratios)
    # bilinear_interpolation(target, )
    check_interpolation()
