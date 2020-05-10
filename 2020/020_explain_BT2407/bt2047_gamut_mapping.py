# -*- coding: utf-8 -*-
"""
BT2407 実装用の各種LUTを作成する
===============================

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from multiprocessing import Pool, cpu_count, Array


# import my libraries
import bt2407_parameters as btp
import color_space as cs
from cielab import bilinear_interpolation


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def get_chroma_lightness_val_specfic_hue(
        hue=30/360*2*np.pi,
        lh_lut_name=btp.get_gamut_boundary_lut_name(cs.BT709)):
    lh_lut = np.load(lh_lut_name)
    lstar = np.linspace(0, 100, lh_lut.shape[0])
    hue_list = np.ones((lh_lut.shape[1])) * hue
    lh = np.dstack([lstar, hue_list])
    chroma = bilinear_interpolation(lh, lh_lut)

    return np.dstack((chroma, lstar))[0]


def calc_chroma_lightness_using_length_from_l_focal(
        distance, degree, l_focal):
    """
    L_Focal からの距離(distance)から chroma, lightness 値を
    三角関数の計算で算出する。

    Parameters
    ----------
    distance : array_like
        Chroma-Lightness 平面における L_focal からの距離の配列。
        例えば Lightness Mapping 後の距離が入ってたりする。
    degree : array_like
        L_focal からの角度。Chroma軸が0°である。
    l_focal : array_like
        l_focal の配列。Lightness の値のリスト。

    Returns
    ------
    chroma : array_like
        Chroma値
    lightness : array_like
        Lightness値
    """
    chroma = distance * np.cos(degree)
    lightness = distance * np.sin(degree) + l_focal

    return chroma, lightness


def calc_chroma_lightness_using_length_from_c_focal(
        distance, degree, c_focal):
    """
    C_Focal からの距離(distance)から chroma, lightness 値を
    三角関数の計算で算出する。

    Parameters
    ----------
    distance : array_like
        Chroma-Lightness 平面における C_focal からの距離の配列。
        例えば Lightness Mapping 後の距離が入ってたりする。
    degree : array_like
        C_focal からの角度。Chroma軸が0°である。
    c_focal : array_like
        c_focal の配列。Chroma の値のリスト。

    Returns
    ------
    chroma : array_like
        Chroma値
    lightness : array_like
        Lightness値
    """
    chroma = distance * np.cos(degree) + c_focal
    lightness = distance * np.sin(degree)

    return chroma, lightness


def main_func():
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
