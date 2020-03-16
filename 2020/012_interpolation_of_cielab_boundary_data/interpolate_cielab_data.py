# -*- coding: utf-8 -*-
"""
CIELAB 
==============

## 方針

HUE を求める。HUE を360 で割って正規化。
roundup(HUE * sample_num), rounddown(HUE * sample_num) で補間対象のindexが求まる？
L* 方向も同様で

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

    """
    l_sample_num = lut2d.shape[0]
    h_sample_num = lut2d.shape[1]
    indices, ratios = calc_bilinear_sample_data(
        lh, l_sample_num, h_sample_num)

    # interpolation in Hue direction
    temp_hi = lut2d[indices[..., 0], indices[..., 2]] * (1 - ratios[..., 1])\
        + lut2d[indices[..., 0], indices[..., 3]] * ratios[..., 1]
    temp_lo = lut2d[indices[..., 1], indices[..., 2]] * (1 - ratios[..., 1])\
        + lut2d[indices[..., 1], indices[..., 3]] * ratios[..., 1]

    # interpolation in Luminance direction
    result = temp_hi * (1 - ratios[..., 0]) + temp_lo * ratios[..., 0]

    return result[0]


def check_interpolation():
    """"""
    chroma_lut = np.load("./boundary_data/Chroma_BT709_l_256_h_256.npy")
    h_idx = np.arange(0, 256, 8)
    l_idx = np.arange(0, 256, 64)
    chroma2 = chroma_lut[l_idx][:, h_idx]
    print(chroma2.shape)

    l_sample_num, h_sample_num = chroma2.shape

    l_val = 40
    hhh_num = 256
    ll = l_val * np.ones(hhh_num)
    hue = np.linspace(0, 2 * np.pi, hhh_num)
    lh = np.dstack((ll, hue))
    chroma_interpolation = bilinear_interpolation(lh, chroma2)
    print(chroma_interpolation.shape)

    l_temp = l_val / 100 * (l_sample_num - 1)
    l_lo_idx = int(np.floor(l_temp))
    l_hi_idx = int(np.ceil(l_temp))
    print(l_temp, l_lo_idx, l_hi_idx)

    rad = np.linspace(0, 2 * np.pi, h_sample_num)
    rad2 = np.linspace(0, 2 * np.pi, hhh_num)
    a_lo = chroma2[l_lo_idx] * np.cos(rad)
    b_lo = chroma2[l_lo_idx] * np.sin(rad)
    a_hi = chroma2[l_hi_idx] * np.cos(rad)
    b_hi = chroma2[l_hi_idx] * np.sin(rad)
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
    # plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    # print("plot l_idx={}".format(idx))
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
