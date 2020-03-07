#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# Color Space モジュール

## 概要
Primaries および White Point の管理。
以下の情報を取れたりする。

* White Point
* Primaries

## 設計思想
In-Out は原則 [0:1] のレンジで行う。
別途、Global変数として最大輝度をパラメータとして持ち、
輝度⇔信号レベルの相互変換はその変数を使って行う。
"""

import os
import numpy as np
from colour.colorimetry import ILLUMINANTS
from colour import RGB_COLOURSPACES
from colour.models import xy_to_XYZ
from colour.adaptation import chromatic_adaptation_matrix_VonKries as cat02_mtx
from scipy import linalg

# Define
CMFS_NAME = 'CIE 1931 2 Degree Standard Observer'
D65 = ILLUMINANTS[CMFS_NAME]['D65']

# NAME
BT709 = 'ITU-R BT.709'
BT2020 = 'ITU-R BT.2020'
ACES_AP0 = 'ACES2065-1'
ACES_AP1 = 'ACEScg'
S_GAMUT3 = 'S-Gamut3'
S_GAMUT3_CINE = 'S-Gamut3.Cine'
ALEXA_WIDE_GAMUT = 'ALEXA Wide Gamut'
V_GAMUT = 'V-Gamut'
CINEMA_GAMUT = 'Cinema Gamut'
RED_WIDE_GAMUT_RGB = 'REDWideGamutRGB'
DCI_P3 = 'DCI-P3'
SRTB = 'sRGB'
P3_D65 = 'P3-D65'


def xy_to_xyz_internal(xy):
    rz = 1 - (xy[0][0] + xy[0][1])
    gz = 1 - (xy[1][0] + xy[1][1])
    bz = 1 - (xy[2][0] + xy[2][1])

    xyz = [[xy[0][0], xy[0][1], rz],
           [xy[1][0], xy[1][1], gz],
           [xy[2][0], xy[2][1], bz]]

    return xyz


def calc_rgb_to_xyz_matrix(gamut_xy, white_large_xyz):
    """
    RGB2XYZ Matrixを計算する

    Parameters
    ----------
    gamut_xy : ndarray
        gamut. shape should be (3, 2).
    white_large_xyz : ndarray
        large xyz value like [95.047, 100.000, 108.883].

    Returns
    -------
    array_like
        a cms pattern image.

    """

    # まずは xyz 座標を準備
    # ------------------------------------------------
    if np.array(gamut_xy).shape == (3, 2):
        gamut = xy_to_xyz_internal(gamut_xy)
    elif np.array(gamut_xy).shape == (3, 3):
        gamut = gamut_xy.copy()
    else:
        raise ValueError("invalid xy gamut parameter.")

    gamut_mtx = np.array(gamut)

    # 白色点の XYZ を算出。Y=1 となるように調整
    # ------------------------------------------------
    large_xyz = [white_large_xyz[0] / white_large_xyz[1],
                 white_large_xyz[1] / white_large_xyz[1],
                 white_large_xyz[2] / white_large_xyz[1]]
    large_xyz = np.array(large_xyz)

    # Sr, Sg, Sb を算出
    # ------------------------------------------------
    s = linalg.inv(gamut_mtx[0:3]).T.dot(large_xyz)

    # RGB2XYZ 行列を算出
    # ------------------------------------------------
    s_matrix = [[s[0], 0.0,  0.0],
                [0.0,  s[1], 0.0],
                [0.0,  0.0,  s[2]]]
    s_matrix = np.array(s_matrix)
    rgb2xyz_mtx = gamut_mtx.T.dot(s_matrix)

    return rgb2xyz_mtx


def get_rgb_to_xyz_matrix(name):
    """
    RGB to XYZ の Matrix を求める。
    DCI-P3 で D65 の係数を返せるように内部関数化した。
    """
    if name != "DCI-P3":
        rgb_to_xyz_matrix = RGB_COLOURSPACES[name].RGB_to_XYZ_matrix
    else:
        rgb_to_xyz_matrix\
            = calc_rgb_to_xyz_matrix(RGB_COLOURSPACES[DCI_P3].primaries,
                                     xy_to_XYZ(ILLUMINANTS[CMFS_NAME]['D65']))

    return rgb_to_xyz_matrix


def get_xyz_to_rgb_matrix(name):
    """
    XYZ to RGB の Matrix を求める。
    DCI-P3 で D65 の係数を返せるように内部関数化した。
    """
    if name != "DCI-P3":
        xyz_to_rgb_matrix = RGB_COLOURSPACES[name].XYZ_to_RGB_matrix
    else:
        rgb_to_xyz_matrix\
            = calc_rgb_to_xyz_matrix(RGB_COLOURSPACES[DCI_P3].primaries,
                                     xy_to_XYZ(ILLUMINANTS[CMFS_NAME]['D65']))
        xyz_to_rgb_matrix = linalg.inv(rgb_to_xyz_matrix)

    return xyz_to_rgb_matrix


def get_white_point(name=ACES_AP0):
    if name == "DCI-P3":
        return ILLUMINANTS[CMFS_NAME]['D65']
    else:
        illuminant = RGB_COLOURSPACES[name].illuminant
        return ILLUMINANTS[CMFS_NAME][illuminant]


def rgb2rgb_mtx(src_name, dst_name):
    src_white = xy_to_XYZ(get_white_point(src_name))
    dst_white = xy_to_XYZ(get_white_point(dst_name))

    chromatic_adaptation_mtx = cat02_mtx(src_white, dst_white, 'CAT02')
    src_rgb2xyz_mtx = get_rgb_to_xyz_matrix(src_name)
    dst_xyz2rgb_mtx = get_xyz_to_rgb_matrix(dst_name)

    temp = np.dot(chromatic_adaptation_mtx, src_rgb2xyz_mtx)
    mtx = np.dot(dst_xyz2rgb_mtx, temp)

    return mtx


def mtx44_from_mtx33(mtx):
    out_mtx = [[mtx[0][0], mtx[0][1], mtx[0][2], 0],
               [mtx[1][0], mtx[1][1], mtx[1][2], 0],
               [mtx[2][0], mtx[2][1], mtx[2][2], 0],
               [0, 0, 0, 1]]

    return np.array(out_mtx)


def ocio_matrix_transform_mtx(src_name, dst_name):
    """
    OpenColorIO の MatrixTransform に食わせる Matrix を吐く。
    """
    mtx33 = rgb2rgb_mtx(src_name, dst_name)
    mtx44 = mtx44_from_mtx33(mtx33)

    return mtx44.flatten().tolist()


def get_primaries(color_space_name=BT709):
    return RGB_COLOURSPACES[color_space_name].primaries


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # print(rgb2rgb_mtx(DCI_P3, ACES_AP0))
    # print(rgb2rgb_mtx(BT709, DCI_P3))
    # ocio_config_mtx_str(ACES_AP0, DCI_P3)
    # ocio_config_mtx_str(DCI_P3, ACES_AP0)
    # print(get_white_point(SRTB))
    # print(get_xyz_to_rgb_matrix(SRTB))
    # bt709_ap0 = rgb2rgb_mtx(BT709, ACES_AP0)
    # print(bt709_ap0)
    # ap0_bt709 = rgb2rgb_mtx(ACES_AP0, BT709)
    # print(ocio_matrix_transform_mtx(ACES_AP0, BT709))
    # print(ocio_matrix_transform_mtx(BT709, ACES_AP0))

    print(ocio_matrix_transform_mtx(ACES_AP0, DCI_P3))
    print(ocio_matrix_transform_mtx(DCI_P3, ACES_AP0))
