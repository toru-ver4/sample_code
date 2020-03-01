#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# 概要
光に関するモジュール

# 使い方

# references
these data have been downloaded from following site.
[Munsell Color Science Laboratory](https://www.rit.edu/cos/colorscience/rc_useful_data.php)

"""

import os
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


def color_temp_to_small_xy(temperature):
    """
    # 概要
    色温度から xy座標を計算する

    # 注意事項
    temperature は numpy であること。1次元。
    """
    x_shita = 0.244063 + 0.09911 * (10 ** 3) / (temperature ** 1)\
        + 2.9678 * (10 ** 6) / (temperature ** 2)\
        - 4.607 * (10 ** 9) / (temperature ** 3)
    x_ue = 0.237040 + 0.24748 * (10 ** 3) / (temperature ** 1)\
        + 1.9018 * (10 ** 6) / (temperature ** 2)\
        - 2.0064 * (10 ** 9) / (temperature ** 3)
    x = x_ue * (temperature > 7000) + x_shita * (temperature <= 7000)
    y = -3.0000 * (x ** 2) + 2.870 * x - 0.275

    return x, y


def _get_d_illuminants_s_coef():
    """
    # 概要
    D光源の算出に必要な係数(S0, S1, S2)を取得する
    """

    filename = os.path.dirname(os.path.abspath(__file__))\
        + os.path.normpath("/data/d_illuminant_s0s1s2_components.csv")
    data = np.loadtxt(filename, delimiter=',', skiprows=1).T

    return data


def _get_d_illuminants_m_coef(x, y):
    """
    # 概要
    D光源の算出に必要な係数(M1, M2)を取得する

    # 注意事項
    x, y は numpy であること。1次元。
    """
    m1 = (-1.3515 - 1.7703 * x + 5.9114 * y) / (0.0241 + 0.2562 * x - 0.7341 * y)
    m2 = (0.0300 - 31.4424 * x + 30.0717 * y) / (0.0241 + 0.2562 * x - 0.7341 * y)

    return m1, m2


def get_d_illuminants_spectrum(temperature):
    """
    # 概要
    スペクトルを求める

    # 注意事項
    temperature は numpy であること。1次元。
    """
    s_param = _get_d_illuminants_s_coef()
    wavelength = np.uint16(s_param[0])
    s0 = s_param[1]
    s1 = s_param[2]
    s2 = s_param[3]
    x, y = color_temp_to_small_xy(temperature)
    m1, m2 = _get_d_illuminants_m_coef(x, y)
    s = []
    for idx, t in enumerate(temperature):
        s.append(s0 + m1[idx] * s1 + m2[idx] * s2)
    s = np.array(s)

    return wavelength, s


def get_d65_spectrum():
    """
    # brief
    return (wavelength, spectrum) pair.
    # source
    Selected Colorimetric Tables(cie)
    http://www.cie.co.at/index.php/LEFTMENUE/DOWNLOADS
    """

    filename = os.path.dirname(os.path.abspath(__file__))\
        + os.path.normpath("/data/d65_spectrum.csv")
    data = np.loadtxt(filename, delimiter=',', skiprows=1).T

    return np.uint16(data[0]), data[1]


def get_cie1931_color_matching_function():
    """
    # brief
    return (wavelength, spectrum) pair.
    # src
    Selected Colorimetric Tables(cie)
    http://www.cie.co.at/index.php/LEFTMENUE/DOWNLOADS
    """

    filename = os.path.dirname(os.path.abspath(__file__))\
        + os.path.normpath("/data/cie_1931_color_matching_function.csv")
    data = np.loadtxt(filename, delimiter=',', skiprows=1).T

    return np.uint16(data[0]), data[1:]


if __name__ == '__main__':
    t = np.arange(4000, 10100, 100, dtype=np.float64)
    x, y = color_temp_to_small_xy(t)
    data = _get_d_illuminants_s_coef()
    m1, m2 = _get_d_illuminants_m_coef(x, y)
    wl, s = get_d_illuminants_spectrum(t)
    print(wl.shape)
    print(s[10].shape)
    wl1, s = get_d65_spectrum()
    wl2, xyz = get_cie1931_color_matching_function()
    idx = np.in1d(wl1, wl2)
