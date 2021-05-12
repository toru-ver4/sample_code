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
from colour.colorimetry import CCS_ILLUMINANTS as ILLUMINANTS
from colour import RGB_COLOURSPACES
from colour.models import xy_to_XYZ
from colour import xy_to_xyY, xyY_to_XYZ, XYZ_to_RGB, RGB_to_XYZ, XYZ_to_Lab,\
    Lab_to_XYZ
from colour.adaptation import matrix_chromatic_adaptation_VonKries as cat02_mtx
from scipy import linalg

# Define
CMFS_NAME = 'CIE 1931 2 Degree Standard Observer'
D65 = ILLUMINANTS[CMFS_NAME]['D65']
D50 = ILLUMINANTS[CMFS_NAME]['D50']

D65_XYZ = xyY_to_XYZ(xy_to_xyY(D65))
D50_XYZ = xyY_to_XYZ(xy_to_xyY(D50))

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


def calc_rgb_from_xyY(xyY, color_space_name, white=D65):
    """
    calc rgb from xyY.

    Parameters
    ----------
    xyY : ndarray
        xyY values.
    color_space_name : str
        the name of the target color space.
    white : ndarray
        white point. ex: np.array([0.3127, 0.3290])

    Returns
    -------
    ndarray
        rgb linear value (not clipped, so negative values may be present).

    Examples
    --------
    >>> xyY = np.array(
    ...     [[0.3127, 0.3290, 1.0], [0.64, 0.33, 0.2], [0.30, 0.60, 0.6]])
    >>> calc_rgb_from_xyY(
    ...     xyY=xyY, color_space_name=cs.BT709, white=D65)
    [[  1.00000000e+00   1.00000000e+00   1.00000000e+00]
     [  9.40561207e-01   1.66533454e-16  -1.73472348e-17]
     [ -2.22044605e-16   8.38962916e-01  -6.93889390e-18]]
    """
    rgb = calc_rgb_from_XYZ(
        xyY_to_XYZ(xyY), color_space_name=color_space_name, white=white)

    return rgb


def calc_rgb_from_XYZ(XYZ, color_space_name, white=D65):
    """
    calc rgb from XYZ.

    Parameters
    ----------
    XYZ : ndarray
        XYZ values.
    color_space_name : str
        the name of the target color space.
    white : ndarray
        white point. ex: np.array([0.3127, 0.3290])

    Returns
    -------
    ndarray
        rgb linear value (not clipped, so negative values may be present).

    Examples
    --------
    >>> xyY = np.array(
    ...     [[0.3127, 0.3290, 1.0], [0.64, 0.33, 0.2], [0.30, 0.60, 0.6]])
    >>> calc_rgb_from_XYZ(
    ...     XYZ=xyY_to_XYZ(xyY), color_space_name=cs.BT709, white=D65)
    [[  1.00000000e+00   1.00000000e+00   1.00000000e+00]
     [  9.40561207e-01   1.66533454e-16  -1.73472348e-17]
     [ -2.22044605e-16   8.38962916e-01  -6.93889390e-18]]
    """
    rgb = XYZ_to_RGB(
        XYZ, white, white,
        RGB_COLOURSPACES[color_space_name].XYZ_to_RGB_matrix)

    return rgb


def calc_XYZ_from_rgb(rgb, color_space_name, white=D65):
    """
    calc XYZ from rgb.

    Parameters
    ----------
    rgb : ndarray
        rgb linear values.
    color_space_name : str
        the name of the target color space.
    white : ndarray
        white point. ex: np.array([0.3127, 0.3290])

    Returns
    -------
    ndarray
        rgb linear value (not clipped, so negative values may be present).

    Examples
    --------
    >>> rgb = np.array(
    ...     [[1.0, 1.0, 1.0], [0.18, 0.18, 0.18], [1.0, 0.0, 0.0]])
    >>> XYZ = calc_XYZ_from_rgb(
    ...     rgb=rgb, color_space_name=cs.BT709, white=D65)
    >>> XYZ_to_xyY(XYZ)
    [[ 0.3127      0.329       1.        ]
     [ 0.3127      0.329       0.18      ]
     [ 0.64        0.33        0.21263901]]
    """
    XYZ = RGB_to_XYZ(
        rgb, white, white,
        RGB_COLOURSPACES[color_space_name].RGB_to_XYZ_matrix)

    return XYZ


def split_tristimulus_values(data):
    """
    Examples
    --------
    >>> data = np.array(
    ...     [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> split_tristimulus_values(data)
    (array([1, 4, 7]), array([2, 5, 8]), array([3, 6, 9]))
    """
    x0 = data[..., 0]
    x1 = data[..., 1]
    x2 = data[..., 2]

    return x0, x1, x2


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


def lab_to_rgb(lab, color_space_name, xyz_white=D65, rgb_white=D65):
    rgb_linear = large_xyz_to_rgb(
        xyz=Lab_to_XYZ(lab), color_space_name=color_space_name,
        xyz_white=xyz_white, rgb_white=rgb_white)

    return rgb_linear


def large_xyz_to_rgb(
        xyz, color_space_name, xyz_white=D65, rgb_white=D65):
    rgb_linear = XYZ_to_RGB(
        xyz, xyz_white, rgb_white,
        RGB_COLOURSPACES[color_space_name].matrix_XYZ_to_RGB)

    return rgb_linear


def rgb_to_large_xyz(
        rgb, color_space_name, rgb_white=D65, xyz_white=D65):
    large_xyz = RGB_to_XYZ(
        rgb, rgb_white, xyz_white,
        RGB_COLOURSPACES[color_space_name].matrix_RGB_to_XYZ)

    return large_xyz


def rgb_to_lab(
        rgb, color_space_name, rgb_white=D65, xyz_white=D65):
    large_xyz = rgb_to_large_xyz(
        rgb=rgb, color_space_name=color_space_name,
        rgb_white=rgb_white, xyz_white=xyz_white)
    lab = XYZ_to_Lab(large_xyz)

    return lab


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

    # print(ocio_matrix_transform_mtx(ACES_AP0, DCI_P3))
    # print(ocio_matrix_transform_mtx(DCI_P3, ACES_AP0))

    # xyY = np.array(
    #     [[0.3127, 0.3290, 1.0], [0.64, 0.33, 0.2], [0.30, 0.60, 0.6]])
    # result = calc_rgb_from_xyY(
    #     xyY=xyY, color_space_name=BT709, white=D65)
    # print(result)

    # xyY = np.array(
    #     [[0.3127, 0.3290, 1.0], [0.64, 0.33, 0.2], [0.30, 0.60, 0.6]])
    # result = calc_rgb_from_XYZ(
    #     XYZ=xyY_to_XYZ(xyY), color_space_name=BT709, white=D65)
    # print(result)

    # data = np.array(
    #     [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(split_tristimulus_values(data))

    # from colour import XYZ_to_xyY
    # rgb = np.array(
    #     [[1.0, 1.0, 1.0], [0.18, 0.18, 0.18], [1.0, 0.0, 0.0]])
    # XYZ = calc_XYZ_from_rgb(
    #     rgb=rgb, color_space_name=BT709, white=D65)
    # print(XYZ_to_xyY(XYZ))
