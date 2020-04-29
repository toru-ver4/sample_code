# -*- coding: utf-8 -*-
"""
Lightness Mapping の実行
=====================================

"""

# import standard libraries
import os
import ctypes

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from colour import Lab_to_XYZ, XYZ_to_RGB
from colour.models import BT2020_COLOURSPACE
from multiprocessing import Pool, cpu_count, Array

# import my libraries
import plot_utility as pu
import make_cusp_focal_lut as mcfl
import interpolate_cielab_data as icd
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


CHROMA_MAP_LUT_DEGREE_SAMPLE_MAX = 1024
CHROMA_MAP_LUT_HUE_SAMPLE_MAX = 1024

CHROMA_MAP_LUT_DEGREE_SAMPLE = 256

CHROMA_MAP_LUT_LUMINANCE_NAME = "./CHROMA_MAP_L.npy"
CHROMA_MAP_LUT_CHROMA_NAME = "./CHROMA_MAP_C.npy"


def linspace_2d(st, ed, sample):
    """
    st, ed が1次元行だった場合に一気に2次元配列を作る。
    y = ax + b 形式で計算する。具体例は Example 参照。

    Examples
    --------
    >>> st = np.array([0, 2, 4])
    >>> ed = np.array([3, 8, 16])
    >>> linspace_2d(st, ed, sample=4)
    >>> [[0, 1, 2, 3],
    ...  [2, 4, 6, 8],
    ...  [4, 8, 12, 16]]
    """
    a = (ed - st)
    b = st
    x = np.linspace(0, 1, sample)
    out = a[:, np.newaxis] * x[np.newaxis, :] + b[:, np.newaxis]

    return out


def calc_chroma_map_degree(l_focal, c_focal):
    """
    Chroma Mapping の Destination の位置用の LUT の
    Start, End の Degree を計算。
    """
    st_degree_l = -np.arctan(l_focal/c_focal)
    ed_degree_l = np.pi/2 * np.ones_like(st_degree_l)
    st_degree_c = np.pi + st_degree_l
    ed_degree_c = np.pi * np.ones_like(st_degree_c)

    return st_degree_l, ed_degree_l, st_degree_c, ed_degree_c


def get_chroma_lightness_val_specfic_hue(
        hue=30/360*2*np.pi, lh_lut_name=mcfl.BT709_BOUNDARY):
    lh_lut = np.load(lh_lut_name)
    lstar = np.linspace(0, 100, lh_lut.shape[0])
    hue_list = np.ones((lh_lut.shape[1])) * hue
    lh = np.dstack([lstar, hue_list])
    chroma = icd.bilinear_interpolation(lh, lh_lut)

    return np.dstack((chroma, lstar))[0]


def make_src_cl_value(src_sample, cl_outer):
    """
    Lightness Mapping のテストとして、Outer Gamut の
    Boundary を Lightness 方向にサンプリングして
    src data を作る。src data は [Chroma, Lightness] が
    入ってるデータ。
    """
    f = interpolate.interp1d(cl_outer[..., 1], cl_outer[..., 0])
    src_l = np.linspace(0, 100, src_sample)
    src_c = f(src_l)
    src_cl = np.dstack((src_c, src_l))[0]

    return src_cl


def convert_cl_value_to_rgb_gamma24(cl_value, hue):
    """
    Lightness Mapping に利用する Chroma-Lightness データを
    Gamma2.4 の RGBデータに変換する。
    なお、このRGB値は Matplotlib 用の暫定値である。
    正しくない。(XYZ to RGB に BT.2020 の係数を使うから)
    """
    lstar = cl_value[..., 1]
    aa = np.cos(hue) * cl_value[..., 0]
    bb = np.sin(hue) * cl_value[..., 0]

    lab = np.dstack((lstar, aa, bb))
    large_xyz = Lab_to_XYZ(lab, tpg.D65_WHITE)
    linear_rgb = XYZ_to_RGB(
        large_xyz, tpg.D65_WHITE, tpg.D65_WHITE,
        BT2020_COLOURSPACE.XYZ_to_RGB_matrix)
    rgb = np.clip(linear_rgb, 0.0, 1.0) ** (1/2.4)

    return rgb


def _debug_plot_lightness_mapping_specific_hue(
        hue, cl_inner, cl_outer, lcusp, inner_cusp, outer_cusp,
        src_cl_value, dst_cl_value, l_cusp, l_focal, c_focal):
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(14, 8),
        graph_title=f"HUE = {hue/2/np.pi*360:.1f}°",
        graph_title_size=None,
        xlabel="Chroma",
        ylabel="Lightness",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=[-3, 103],
        xtick=None,
        ytick=[x * 10 for x in range(11)],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.patch.set_facecolor("#E0E0E0")
    in_color = "#707070"
    ou_color = "#000000"
    fo_color = "#A0A0A0"

    # gamut boundary
    ax1.plot(
        cl_inner[..., 0], cl_inner[..., 1], '--', c=in_color, label="BT.709")
    ax1.plot(cl_outer[..., 0], cl_outer[..., 1], c=ou_color, label="BT.2020")

    # gamut cusp
    ax1.plot(inner_cusp[1], inner_cusp[0], 's', ms=10, mec=ou_color,
             c=in_color, label="BT.709 Cusp")
    ax1.plot(outer_cusp[1], outer_cusp[0], 's', ms=10, mec=in_color,
             c=ou_color, label="BT.2020 Cusp")

    # l_cusp, l_focal, c_focal
    ax1.plot([0], [l_cusp], 'x', ms=12, mew=4, c=in_color, label="L_cusp")
    ax1.plot([0], [l_focal], 'x', ms=12, mew=4, c=ou_color, label="L_focal")
    ax1.plot([c_focal], [0], '*', ms=12, mew=3, c=ou_color, label="C_focal")
    ax1.plot([0, c_focal], [l_focal, 0], ':', c=fo_color)

    # src, dst data
    src_cl_rgb = convert_cl_value_to_rgb_gamma24(src_cl_value, hue)
    dst_cl_rgb = convert_cl_value_to_rgb_gamma24(dst_cl_value, hue)
    ax1.scatter(src_cl_value[..., 0], src_cl_value[..., 1], s=150,
                c=src_cl_rgb[0], label="src", zorder=3)
    ax1.scatter(dst_cl_value[..., 0], dst_cl_value[..., 1], s=150,
                c=dst_cl_rgb[0], label="dst", zorder=3)

    # annotation settings
    arrowprops = dict(
        facecolor='#000000', shrink=0.0, headwidth=8, headlength=10,
        width=1, alpha=0.8)
    for idx in range(len(src_cl_value)):
        color = (1 - np.max(src_cl_rgb[0, idx]))
        arrowprops['facecolor'] = np.array((color, color, color))
        ax1.annotate(
            "", xy=(dst_cl_value[idx, 0], dst_cl_value[idx, 1]),
            xytext=(src_cl_value[idx, 0], src_cl_value[idx, 1]),
            xycoords='data', textcoords='data', ha='left', va='bottom',
            arrowprops=arrowprops)
    graph_name = f"./figure/lm_test_HUE_{hue/2/np.pi*360:.1f}.png"
    plt.legend(loc='upper right')
    # plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(graph_name)  # オプション付けるとエラーになるので外した
    plt.show()


def calc_value_from_hue_1dlut(val, lut):
    x = np.linspace(0, 2*np.pi, len(lut))
    f = interpolate.interp1d(x, lut)
    y = f(val)

    return y


def lightness_mapping_to_l_focal(l_out, c_out, c_map, l_focal):
    return ((l_out - l_focal) * c_map) / c_out + l_focal


def lightness_mapping_from_c_focal(l_out, c_out, c_map, c_focal):
    return (l_out * (c_out - c_map)) / (c_focal - c_out) + l_out


def _lightness_mapping_trial(src_cl, l_focal, c_focal):
    """
    とりあえず Lightness Mapping の雰囲気を確認する用の実装

    * Cmap は一律で Cout * 0.5 に設定してみる

    Parameters
    ----------
    src_cl : array_like (2d array, shape is N x 2)
        target Chroma, Lightness value.
        src_cl[..., 0]: Chroma
        src_cl[..., 1]: Lightness
    l_focal : float
        L_focal value

    Returns
    -------
    array_like (2d array, shape is N x 2)
        Chroma-Ligheness Value after lightness mapping.
    """
    l_out = src_cl[..., 1]
    c_out = src_cl[..., 0]
    c_map = c_out * 0.7

    # どっちの数式を使うかの判定
    l_map = np.where(
        l_out >= (-l_focal * c_out) / c_focal + l_focal,
        lightness_mapping_to_l_focal(l_out, c_out, c_map, l_focal),
        lightness_mapping_from_c_focal(l_out, c_out, c_map, c_focal)
    )

    dst_cl = np.dstack((c_map, l_map))[0]

    return dst_cl


def _try_lightness_mapping_specific_hue(hue=30/360*2*np.pi):
    cl_inner = get_chroma_lightness_val_specfic_hue(hue, mcfl.BT709_BOUNDARY)
    cl_outer =\
        get_chroma_lightness_val_specfic_hue(hue, mcfl.BT2020_BOUNDARY)

    # cusp 準備
    lh_inner_lut = np.load(mcfl.BT709_BOUNDARY)
    lh_outer_lut = np.load(mcfl.BT2020_BOUNDARY)
    lcusp = mcfl.calc_l_cusp_specific_hue(hue, lh_inner_lut, lh_outer_lut)
    inner_cusp = mcfl.calc_cusp_in_lc_plane(hue, lh_inner_lut)
    outer_cusp = mcfl.calc_cusp_in_lc_plane(hue, lh_outer_lut)

    # l_cusp, l_focal, c_focal 準備
    l_cusp_lut = np.load(mcfl.L_CUSP_NAME)
    l_focal_lut = np.load(mcfl.L_FOCAL_NAME)
    c_focal_lut = np.load(mcfl.C_FOCAL_NAME)
    l_cusp = calc_value_from_hue_1dlut(hue, l_cusp_lut)
    l_focal = calc_value_from_hue_1dlut(hue, l_focal_lut)
    c_focal = calc_value_from_hue_1dlut(hue, c_focal_lut)

    # テストデータ準備
    src_cl_value = make_src_cl_value(src_sample=11, cl_outer=cl_outer)
    dst_cl_value = _lightness_mapping_trial(src_cl_value, l_focal, c_focal)

    _debug_plot_lightness_mapping_specific_hue(
        hue, cl_inner, cl_outer, lcusp, inner_cusp, outer_cusp,
        src_cl_value, dst_cl_value, l_cusp, l_focal, c_focal)


def _calc_ab_coef_from_lfocal_and_cl_src(cl_src, l_focal):
    """
    chroma-lightness のサンプル点から l_focal へと伸びる直線
    y=ax+b の各係数 a, b を求める。
    """
    x2 = cl_src[..., 0]
    y2 = cl_src[..., 1]
    x1 = 0
    y1 = l_focal

    a = np.where(
        x1 != x2,
        (y2 - y1) / (x2 - x1),
        0
    )
    # b = l_focal * np.ones_like(a)
    b = y2 - a * x2

    return a, b


def _calc_ab_coef_from_cfocal_and_cl_src(cl_src, c_focal):
    """
    chroma-lightness のサンプル点から c_focal へと伸びる直線
    y=ax+b の各係数 a, b を求める。
    """
    x2 = cl_src[..., 0]
    y2 = cl_src[..., 1]
    x1 = c_focal
    y1 = 0

    a = np.where(
        x1 != x2,
        (y2 - y1) / (x2 - x1),
        0
    )
    b = y2 - a * x2

    return a, b


def _calc_ab_coef_from_cl_point(cl_point):
    """
    y=ax+b の a, b を求める。
    直線は隣接し合う2点を結ぶ線である。
    """
    x_list = cl_point[..., 0]
    y_list = cl_point[..., 1]

    a = (y_list[1:] - y_list[:-1]) / (x_list[1:] - x_list[:-1])
    b = y_list[1:] - a * x_list[1:]

    return a, b


def _debug_plot_ab_for_line(a, b, cl_src):
    x = np.linspace(0, 220, 256)
    y = a[:, np.newaxis] * x[np.newaxis, :] + b[:, np.newaxis]

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Debug Plot",
        graph_title_size=None,
        xlabel="X Axis Label", ylabel="Y Axis Label",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=(-5, 105),
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3)
    for ii in range(len(a)):
        ax1.plot(x, y[ii], 'k:', lw=1)
    ax1.plot(cl_src[..., 0], cl_src[..., 1], 'ko')
    plt.legend(loc='upper left')
    plt.show()


def _debug_plot_ab_for_cl_plane(a, b, cl_point):
    x = np.linspace(0, 220, 256)
    y = a[:, np.newaxis] * x[np.newaxis, :] + b[:, np.newaxis]

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Debug Plot",
        graph_title_size=None,
        xlabel="Chroma", ylabel="Lightness",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=(-5, 105),
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3)
    for ii in range(len(a)):
        ax1.plot(x, y[ii], 'k:', lw=1)
    ax1.plot(cl_point[..., 0], cl_point[..., 1], 'ko')
    plt.legend(loc='upper left')
    plt.show()


def _debug_plot_intersection(a1, b1, a2, b2, cl_point, cl_src, icn_x, icn_y):
    x = np.linspace(-5, 150, 256)
    y1 = a1[:, np.newaxis] * x[np.newaxis, :] + b1[:, np.newaxis]
    y2 = a2[:, np.newaxis] * x[np.newaxis, :] + b2[:, np.newaxis]

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Debug Plot",
        graph_title_size=None,
        xlabel="Chroma", ylabel="Lightness",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=(-3, 103),
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3)
    for ii in range(len(a1)):
        ax1.plot(x, y1[ii], 'k:', lw=1)
    for ii in range(len(a2)):
        ax1.plot(x, y2[ii], '--', c=pu.BLUE, lw=1)
    ax1.plot(icn_x[1], icn_y[1], 'x', c=pu.ORANGE, ms=15, mew=3)
    ax1.plot(cl_point[..., 0], cl_point[..., 1], 'ko')
    ax1.plot(cl_src[..., 0], cl_src[..., 1], 's', c=pu.RED)
    plt.legend(loc='upper left')
    plt.show()


def _debug_plot_valid_intersection(
        a1, b1, a2, b2, cl_point, cl_src, icn_x, icn_y,
        l_focal, c_focal, hue, focal_type='C_focal'):
    x = np.linspace(-5, c_focal, 256)
    y1 = a1[:, np.newaxis] * x[np.newaxis, :] + b1[:, np.newaxis]
    degree = hue / (2 * np.pi) * 360

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title=f"Debug Plot ({focal_type}, {degree:.1f}°)",
        graph_title_size=None,
        xlabel="Chroma", ylabel="Lightness",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=(-3, 103),
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3)
    for ii in range(len(a1)):
        ax1.plot(x, y1[ii], 'k:', lw=1)
    ax1.plot(cl_point[..., 0], cl_point[..., 1], 'k-', lw=2, label="BT.709")
    ax1.plot(cl_src[..., 0], cl_src[..., 1], 's', ms=13, c=pu.RED,
             label="BT.2020 Sample")
    ax1.plot(icn_x, icn_y, 'x', c=pu.GREEN, ms=13, mew=5)
    ax1.plot([0], [l_focal], 'o', c=pu.BLUE, ms=13, label="L_focal")
    ax1.plot([c_focal], [0], 'o', c=pu.PINK, ms=13, label="C_focal")
    plt.legend(loc='upper right')
    fname = f"./figure/{focal_type}_{degree:.1f}.png"
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
    # plt.show()


def solve_equation_for_intersection(cl_point, a1, b1, a2, b2):
    # 1次元方程式っぽいのを解いて交点を算出
    icn_x = (b1[:, np.newaxis] - b2[np.newaxis, :])\
        / (a2[np.newaxis, :] - a1[:, np.newaxis])
    icn_y = a1[:, np.newaxis] * icn_x + b1[:, np.newaxis]

    # 交点から有効点？を抽出
    # print(icn_x.shape)
    ok_src_idx_x_0 = (icn_x >= cl_point[:-1, 0]) & (icn_x <= cl_point[1:, 0])
    ok_src_idx_x_1 = (icn_x <= cl_point[:-1, 0]) & (icn_x >= cl_point[1:, 0])
    ok_src_idx_x = ok_src_idx_x_0 | ok_src_idx_x_1
    ok_src_idx_y = (icn_y >= cl_point[:-1, 1]) & (icn_y <= cl_point[1:, 1])
    ok_src_idx = ok_src_idx_x & ok_src_idx_y
    ok_dst_idx = np.any(ok_src_idx, axis=-1)
    icn_valid_x = np.zeros((icn_x.shape[0]))
    icn_valid_y = np.zeros((icn_y.shape[0]))
    icn_valid_x[ok_dst_idx] = icn_x[ok_src_idx]
    icn_valid_y[ok_dst_idx] = icn_y[ok_src_idx]

    return icn_valid_x, icn_valid_y


def _calc_intersection_of_gamut_and_lines(
        cl_point, cl_src, l_focal, c_focal, hue):
    """
    gamut boundary と Lines の交点を求める。

    Parameters
    ----------
    cl_point : array_like (2d array)
        gamut boundary data(Chroma, Lightness)
    cl_src : array_like (2d array)
        src chroma-lightness points.
    l_focal : float
        l_focal
    c_focal : float
        c_focal
    hue : float
        hue. unit is radian. for debug plot.
    """
    # 各 sl_src と l_focal を結ぶ直線 y=ax+b の a, b の値を出す
    a1_l, b1_l = _calc_ab_coef_from_lfocal_and_cl_src(cl_src, l_focal)

    # 各 sl_src と c_focal を結ぶ直線 y=ax+b の a, b の値を出す
    a1_c, b1_c = _calc_ab_coef_from_cfocal_and_cl_src(cl_src, c_focal)

    # 各 cl_point の2点間の直線 y=ax+b の a, b の値を出す
    # cl_point = cl_point[::48]
    a2, b2 = _calc_ab_coef_from_cl_point(cl_point)

    # 直線群と直線群の交点を求める。(L_focal)
    icn_valid_x_l, icn_valid_y_l = solve_equation_for_intersection(
        cl_point, a1_l, b1_l, a2, b2)

    # 直線群と直線群の交点を求める。(C_focal)
    icn_valid_x_c, icn_valid_y_c = solve_equation_for_intersection(
        cl_point, a1_c, b1_c, a2, b2)

    # debug plot
    # _debug_plot_ab_for_line(a1, b1, cl_src)
    # _debug_plot_ab_for_cl_plane(a2, b2, cl_point)
    # _debug_plot_intersection(a1, b1, a2, b2, cl_point, cl_src, icn_x, icn_y)
    _debug_plot_valid_intersection(
        a1_l, b1_l, a2, b2, cl_point, cl_src, icn_valid_x_l, icn_valid_y_l,
        l_focal, c_focal, hue, focal_type="L_focal")
    _debug_plot_valid_intersection(
        a1_c, b1_c, a2, b2, cl_point, cl_src, icn_valid_x_c, icn_valid_y_c,
        l_focal, c_focal, hue, focal_type="C_focal")


def _check_calc_cmap_on_lc_plane(hue=30/360*2*np.pi):
    """
    L*C*平面において、Cmap が計算できるか確認する。
    """
    # とりあえず L*C* 平面のポリゴン準備
    cl_inner = get_chroma_lightness_val_specfic_hue(hue, mcfl.BT709_BOUNDARY)
    cl_outer =\
        get_chroma_lightness_val_specfic_hue(hue, mcfl.BT2020_BOUNDARY)

    # cusp 準備
    lh_inner_lut = np.load(mcfl.BT709_BOUNDARY)
    lh_outer_lut = np.load(mcfl.BT2020_BOUNDARY)
    lcusp = mcfl.calc_l_cusp_specific_hue(hue, lh_inner_lut, lh_outer_lut)
    inner_cusp = mcfl.calc_cusp_in_lc_plane(hue, lh_inner_lut)
    outer_cusp = mcfl.calc_cusp_in_lc_plane(hue, lh_outer_lut)

    # l_cusp, l_focal, c_focal 準備
    l_cusp_lut = np.load(mcfl.L_CUSP_NAME)
    l_focal_lut = np.load(mcfl.L_FOCAL_NAME)
    c_focal_lut = np.load(mcfl.C_FOCAL_NAME)
    l_cusp = calc_value_from_hue_1dlut(hue, l_cusp_lut)
    l_focal = calc_value_from_hue_1dlut(hue, l_focal_lut)
    c_focal = calc_value_from_hue_1dlut(hue, c_focal_lut)

    # テストポイントの src_cl_value も準備
    src_cl_value = make_src_cl_value(src_sample=12, cl_outer=cl_outer)

    # CL 平面のboundaryの各サンプルとの交点(C*, L*)を求める
    _calc_intersection_of_gamut_and_lines(
        cl_inner, src_cl_value, l_focal, c_focal, hue)

    # src_cl ごとに線分を作成、polygon との交点を算出

    # 交点から Cmap を決定


def _debug_plot_chroma_map_lut_specific_hue(
        hue, cl_inner, cl_outer, lcusp, inner_cusp, outer_cusp,
        l_cusp, l_focal, c_focal, icn_x, icn_y, focal_type, idx):
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(14, 8),
        graph_title=f"HUE = {hue/2/np.pi*360:.1f}°, for {focal_type}",
        graph_title_size=None,
        xlabel="Chroma",
        ylabel="Lightness",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=[-3, 103],
        xtick=None,
        ytick=[x * 10 for x in range(11)],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.patch.set_facecolor("#E0E0E0")
    in_color = pu.BLUE
    ou_color = pu.RED
    fo_color = "#A0A0A0"

    # gamut boundary
    ax1.plot(
        cl_inner[..., 0], cl_inner[..., 1], c=in_color, label="BT.709")
    ax1.plot(cl_outer[..., 0], cl_outer[..., 1], c=ou_color, label="BT.2020")

    # gamut cusp
    ax1.plot(inner_cusp[1], inner_cusp[0], 's', ms=10, mec='k',
             c=in_color, label="BT.709 Cusp")
    ax1.plot(outer_cusp[1], outer_cusp[0], 's', ms=10, mec='k',
             c=ou_color, label="BT.2020 Cusp")

    # l_cusp, l_focal, c_focal
    ax1.plot([0], [l_cusp], 'x', ms=12, mew=4, c=in_color, label="L_cusp")
    ax1.plot([0], [l_focal], 'x', ms=12, mew=4, c=ou_color, label="L_focal")
    ax1.plot([c_focal], [0], '*', ms=12, mew=3, c=ou_color, label="C_focal")
    ax1.plot([0, c_focal], [l_focal, 0], '--', c=fo_color)

    # intersectionx
    ax1.plot(icn_x, icn_y, 'o', ms=12, label="destination")
    if focal_type == "L_focal":
        for x, y in zip(icn_x, icn_y):
            ax1.plot([0, x], [l_focal, y], ':', c='k')
    elif focal_type == "C_focal":
        for x, y in zip(icn_x, icn_y):
            ax1.plot([c_focal, x], [0, y], ':', c='k')
    else:
        pass

    graph_name = f"./video_src/cmap_lut_{focal_type}_{idx:04d}.png"
    plt.legend(loc='upper right')
    # plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(graph_name)  # オプション付けるとエラーになるので外した
    # plt.show()


def _debug_plot_check_chroma_map_lut_specific_hue(
        hue, cl_inner, cl_outer, lcusp, inner_cusp, outer_cusp,
        l_cusp, l_focal, c_focal, icn_x, icn_y, focal_type, idx):
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(14, 8),
        graph_title=f"HUE = {hue/2/np.pi*360:.1f}°, for {focal_type}",
        graph_title_size=None,
        xlabel="Chroma",
        ylabel="Lightness",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=[-3, 103],
        xtick=None,
        ytick=[x * 10 for x in range(11)],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.patch.set_facecolor("#E0E0E0")
    in_color = pu.BLUE
    ou_color = pu.RED
    fo_color = "#A0A0A0"

    # gamut boundary
    ax1.plot(
        cl_inner[..., 0], cl_inner[..., 1], c=in_color, label="BT.709")
    ax1.plot(cl_outer[..., 0], cl_outer[..., 1], c=ou_color, label="BT.2020")

    # gamut cusp
    ax1.plot(inner_cusp[1], inner_cusp[0], 's', ms=10, mec='k',
             c=in_color, label="BT.709 Cusp")
    ax1.plot(outer_cusp[1], outer_cusp[0], 's', ms=10, mec='k',
             c=ou_color, label="BT.2020 Cusp")

    # l_cusp, l_focal, c_focal
    ax1.plot([0], [l_cusp], 'x', ms=12, mew=4, c=in_color, label="L_cusp")
    ax1.plot([0], [l_focal], 'x', ms=12, mew=4, c=ou_color, label="L_focal")
    ax1.plot([c_focal], [0], '*', ms=12, mew=3, c=ou_color, label="C_focal")
    ax1.plot([0, c_focal], [l_focal, 0], '--', c=fo_color)

    # intersectionx
    ax1.plot(icn_x, icn_y, 'o', ms=12, label="destination")
    if focal_type == "L_focal":
        for x, y in zip(icn_x, icn_y):
            ax1.plot([0, x], [l_focal, y], ':', c='k')
    elif focal_type == "C_focal":
        for x, y in zip(icn_x, icn_y):
            ax1.plot([c_focal, x], [0, y], ':', c='k')
    else:
        pass

    graph_name = f"./video_src/cmap_lut_check_{focal_type}_{idx:04d}.png"
    plt.legend(loc='upper right')
    # plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    # plt.savefig(graph_name)  # オプション付けるとエラーになるので外した
    plt.show()


def make_chroma_map_lut_specific_hue(hue=30/360*2*np.pi, idx=0):
    """
    Lfocal, Cfocal を中心とする放射線状のデータが
    どの Chroma値にマッピングされるかを示すLUTを作る。
    """
    # とりあえず L*C* 平面のポリゴン準備
    cl_inner = get_chroma_lightness_val_specfic_hue(hue, mcfl.BT709_BOUNDARY)
    cl_outer =\
        get_chroma_lightness_val_specfic_hue(hue, mcfl.BT2020_BOUNDARY)

    # cusp 準備
    lh_inner_lut = np.load(mcfl.BT709_BOUNDARY)
    lh_outer_lut = np.load(mcfl.BT2020_BOUNDARY)
    lcusp = mcfl.calc_l_cusp_specific_hue(hue, lh_inner_lut, lh_outer_lut)
    inner_cusp = mcfl.calc_cusp_in_lc_plane(hue, lh_inner_lut)
    outer_cusp = mcfl.calc_cusp_in_lc_plane(hue, lh_outer_lut)

    # l_cusp, l_focal, c_focal 準備
    l_cusp_lut = np.load(mcfl.L_CUSP_NAME)
    l_focal_lut = np.load(mcfl.L_FOCAL_NAME)
    c_focal_lut = np.load(mcfl.C_FOCAL_NAME)
    l_cusp = calc_value_from_hue_1dlut(hue, l_cusp_lut)
    l_focal = calc_value_from_hue_1dlut(hue, l_focal_lut)
    c_focal = calc_value_from_hue_1dlut(hue, c_focal_lut)

    st_degree_l, ed_degree_l, st_degree_c, ed_degree_c =\
        calc_chroma_map_degree(l_focal, c_focal)

    # Lfocal用のサンプル点作成
    degree = np.linspace(
        st_degree_l, ed_degree_l, get_degree_sample_num())
    a1_l = np.tan(degree)
    b1_l = l_focal * np.ones_like(degree)

    # Cfocal用のサンプル点作成
    degree = np.linspace(
        st_degree_c, ed_degree_c, get_degree_sample_num())
    a1_c = np.tan(degree)
    b1_c = -a1_c * c_focal

    # 各 cl_point の2点間の直線 y=ax+b の a, b の値を出す
    # cl_inner = cl_inner[::48]
    a2, b2 = _calc_ab_coef_from_cl_point(cl_inner)

    print(np.rad2deg(hue))
    # 直線群と直線群の交点を求める。(L_focal)
    icn_x_l, icn_y_l = solve_equation_for_intersection(
        cl_inner, a1_l, b1_l, a2, b2)
    # _debug_plot_chroma_map_lut_specific_hue(
    #     hue, cl_inner, cl_outer, lcusp, inner_cusp, outer_cusp,
    #     l_cusp, l_focal, c_focal, icn_x_l, icn_y_l, focal_type="L_focal",
    #     idx=idx)

    # 直線群と直線群の交点を求める。(C_focal)
    icn_x_c, icn_y_c = solve_equation_for_intersection(
        cl_inner, a1_c, b1_c, a2, b2)
    # _debug_plot_chroma_map_lut_specific_hue(
    #     hue, cl_inner, cl_outer, lcusp, inner_cusp, outer_cusp,
    #     l_cusp, l_focal, c_focal, icn_x_c, icn_y_c, focal_type="C_focal",
    #     idx=idx)

    cmap_l = ((icn_x_l) ** 2 + (icn_y_l - l_focal) ** 2) ** 0.5
    cmap_c = ((icn_x_c - c_focal) ** 2 + (icn_y_c) ** 2) ** 0.5

    return cmap_l, cmap_c


def calc_icn_xy_from_l_focal(cmap_lut_val_l, degree_l, l_focal):
    icn_x_l = cmap_lut_val_l * np.cos(degree_l)
    icn_y_l = cmap_lut_val_l * np.sin(degree_l) + l_focal

    return icn_x_l, icn_y_l


def calc_icn_xy_from_c_focal(cmap_lut_val_c, degree_c, c_focal):
    icn_x_c = cmap_lut_val_c * np.cos(degree_c) + c_focal
    icn_y_c = cmap_lut_val_c * np.sin(degree_c)

    return icn_x_c, icn_y_c


def _check_chroma_map_lut_data(h_idx):
    hue = h_idx / (get_hue_sample_num() - 1) * 2 * np.pi

    # とりあえず L*C* 平面のポリゴン準備
    cl_inner = get_chroma_lightness_val_specfic_hue(hue, mcfl.BT709_BOUNDARY)
    cl_outer =\
        get_chroma_lightness_val_specfic_hue(hue, mcfl.BT2020_BOUNDARY)

    # cusp 準備
    lh_inner_lut = np.load(mcfl.BT709_BOUNDARY)
    lh_outer_lut = np.load(mcfl.BT2020_BOUNDARY)
    lcusp = mcfl.calc_l_cusp_specific_hue(hue, lh_inner_lut, lh_outer_lut)
    inner_cusp = mcfl.calc_cusp_in_lc_plane(hue, lh_inner_lut)
    outer_cusp = mcfl.calc_cusp_in_lc_plane(hue, lh_outer_lut)

    # l_cusp, l_focal, c_focal 準備
    l_cusp_lut = np.load(mcfl.L_CUSP_NAME)
    l_focal_lut = np.load(mcfl.L_FOCAL_NAME)
    c_focal_lut = np.load(mcfl.C_FOCAL_NAME)
    l_cusp = calc_value_from_hue_1dlut(hue, l_cusp_lut)
    l_focal = calc_value_from_hue_1dlut(hue, l_focal_lut)
    c_focal = calc_value_from_hue_1dlut(hue, c_focal_lut)

    # Chroma Mapping の距離のデータ
    st_degree_l, ed_degree_l, st_degree_c, ed_degree_c =\
        calc_chroma_map_degree(l_focal, c_focal)
    cmap_lut_c = np.load(CHROMA_MAP_LUT_CHROMA_NAME)
    cmap_lut_l = np.load(CHROMA_MAP_LUT_LUMINANCE_NAME)

    degree_sample = cmap_lut_l.shape[1]
    degree_l = np.linspace(st_degree_l, ed_degree_l, degree_sample)
    degree_c = np.linspace(st_degree_c, ed_degree_c, degree_sample)

    icn_x_l, icn_y_l = calc_icn_xy_from_l_focal(
        cmap_lut_val_l=cmap_lut_l[h_idx], degree_l=degree_l, l_focal=l_focal)
    icn_x_c, icn_y_c = calc_icn_xy_from_c_focal(
        cmap_lut_val_c=cmap_lut_c[h_idx], degree_c=degree_c, c_focal=c_focal)

    _debug_plot_check_chroma_map_lut_specific_hue(
        hue, cl_inner, cl_outer, lcusp, inner_cusp, outer_cusp,
        l_cusp, l_focal, c_focal,
        icn_x=icn_x_l, icn_y=icn_y_l,
        focal_type="L_focal", idx=h_idx)
    _debug_plot_check_chroma_map_lut_specific_hue(
        hue, cl_inner, cl_outer, lcusp, inner_cusp, outer_cusp,
        l_cusp, l_focal, c_focal,
        icn_x=icn_x_c, icn_y=icn_y_c,
        focal_type="C_focal", idx=h_idx)


def thread_wrapper_chroa_map_lut_specific_hue(args):
    make_chroma_map_lut_specific_hue(**args)


def get_hue_sample_num(gamut_boundary_lut_name=mcfl.BT709_BOUNDARY):
    """
    shape[0]: luminance sample
    shape[1]: hue sample
    """
    return np.load(gamut_boundary_lut_name).shape[1]


def get_degree_sample_num():
    return CHROMA_MAP_LUT_DEGREE_SAMPLE


def make_chroma_map_lut():
    """
    Lfocal, Cfocal を中心とする放射線状のデータが
    どの Chroma値にマッピングされるかを示すLUTを作る。
    """
    hue_sample = get_hue_sample_num()
    hue_list = np.linspace(0, 2 * np.pi, hue_sample)
    args = []
    cmap_l_buf = []
    cmap_c_buf = []
    for idx, hue in enumerate(hue_list):
        print(np.rad2deg(hue))
        cmap_l, cmap_c = make_chroma_map_lut_specific_hue(hue=hue, idx=idx)
        cmap_l_buf.append(cmap_l)
        cmap_c_buf.append(cmap_c)
        args.append(dict(hue=hue, idx=idx))
        # break
    # with Pool(cpu_count()) as pool:
    #     pool.map(thread_wrapper_chroa_map_lut_specific_hue, args)

    # 整形して .npy で保存
    cmap_l_lut = np.array(cmap_l_buf)
    cmap_c_lut = np.array(cmap_c_buf)

    np.save(CHROMA_MAP_LUT_LUMINANCE_NAME, cmap_l_lut)
    np.save(CHROMA_MAP_LUT_CHROMA_NAME, cmap_c_lut)


def _debug_plot_check_lightness_mapping_specific_hue(
        hue, cl_inner, cl_outer, lcusp, inner_cusp, outer_cusp,
        l_cusp, l_focal, c_focal, x_val, y_val, focal_type, idx=0):
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(14, 8),
        graph_title=f"HUE = {hue/2/np.pi*360:.1f}°, for {focal_type}",
        graph_title_size=None,
        xlabel="Chroma",
        ylabel="Lightness",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=[-33, 143],
        xtick=None,
        ytick=[(x - 3) * 10 for x in range(18)],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.patch.set_facecolor("#E0E0E0")
    in_color = pu.BLUE
    ou_color = pu.RED
    fo_color = "#A0A0A0"
    src_color = pu.GREEN

    # gamut boundary
    ax1.plot(
        cl_inner[..., 0], cl_inner[..., 1], c=in_color, label="BT.709")
    ax1.plot(cl_outer[..., 0], cl_outer[..., 1], c=ou_color, label="BT.2020")

    # gamut cusp
    ax1.plot(inner_cusp[1], inner_cusp[0], 's', ms=10, mec='k',
             c=in_color, label="BT.709 Cusp")
    ax1.plot(outer_cusp[1], outer_cusp[0], 's', ms=10, mec='k',
             c=ou_color, label="BT.2020 Cusp")

    # l_cusp, l_focal, c_focal
    ax1.plot([0], [l_cusp], 'x', ms=12, mew=4, c=in_color, label="L_cusp")
    ax1.plot([0], [l_focal], 'x', ms=12, mew=4, c=ou_color, label="L_focal")
    ax1.plot([c_focal], [0], '*', ms=12, mew=3, c=ou_color, label="C_focal")
    ax1.plot([0, c_focal], [l_focal, 0], '--', c='k')

    # intersectionx
    ax1.plot(x_val, y_val, 'o', ms=12, c=src_color, label="destination")
    if focal_type == "L_focal":
        for x, y in zip(x_val, y_val):
            ax1.plot([0, x], [l_focal, y], ':', c=fo_color)
    elif focal_type == "C_focal":
        for x, y in zip(x_val, y_val):
            ax1.plot([c_focal, x], [0, y], ':', c=fo_color)
    else:
        pass

    graph_name = f"./video_src/lightness_mapping_{focal_type}_{idx:04d}.png"
    plt.legend(loc='upper right')
    plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    # plt.savefig(graph_name)  # オプション付けるとエラーになるので外した
    plt.show()


def _make_debug_luminance_chroma_data_fixed_hue(
        hue, hue_sample_num, st_degree_lut, ed_degree_lut,
        focal_lut, focal_type="l"):
    """
    任意の Hue の Hue-Degree のサンプルデータを作る。
    st_degree, ed_degree を考慮
    """
    sample_num = 7
    hue_idx_low_float = hue / (2 * np.pi) * (hue_sample_num - 1)
    hue_idx_low = int(hue_idx_low_float)
    hue_idx_high = hue_idx_low + 1
    rate = hue_idx_low_float - hue_idx_low

    st_degree = (1 - rate) * st_degree_lut[hue_idx_low]\
        + rate * st_degree_lut[hue_idx_high]
    ed_degree = (1 - rate) * ed_degree_lut[hue_idx_low]\
        + rate * ed_degree_lut[hue_idx_high]


    if focal_type == 'l':
        r1 = np.ones(sample_num) * 30
        r2 = np.ones(sample_num) * 75
        rr = np.append(r1, r2).reshape((2, sample_num))
        degree_data = np.linspace(
            st_degree + (np.sign(st_degree) * np.abs(st_degree) * 0.5),
            ed_degree, sample_num)
        l_focal = calc_value_from_hue_1dlut(hue, focal_lut)
        chroma, lightness = calc_icn_xy_from_l_focal(
            cmap_lut_val_l=rr, degree_l=degree_data, l_focal=l_focal)
    elif focal_type == 'c':
        c_focal = calc_value_from_hue_1dlut(hue, focal_lut)
        r1 = np.ones(sample_num) * c_focal
        r2 = np.ones(sample_num) * (c_focal - 75)
        rr = np.append(r1, r2).reshape((2, sample_num))
        degree_data = np.linspace(
            st_degree - (st_degree * 0.02), ed_degree, sample_num)
        chroma, lightness = calc_icn_xy_from_c_focal(
            cmap_lut_val_c=rr, degree_c=degree_data, c_focal=c_focal)

    return lightness.flatten(), chroma.flatten()


def _make_debug_luminance_chroma_data_fixed_hue2(
        hue, focal_lut, focal_type="l"):
    """
    任意の Hue の Hue-Degree のサンプルデータを作る。
    st_degree, ed_degree を**考慮しない**
    """
    sample_num = 7

    r1 = np.ones(sample_num) * 30
    r2 = np.ones(sample_num) * 70
    rr = np.append(r1, r2).reshape((2, sample_num))

    if focal_type == 'l':
        degree_data = np.linspace(-np.pi/2, np.pi/2, sample_num)
        l_focal = calc_value_from_hue_1dlut(hue, focal_lut)
        chroma, lightness = calc_icn_xy_from_l_focal(
            cmap_lut_val_l=rr, degree_l=degree_data, l_focal=l_focal)
    elif focal_type == 'c':
        degree_data = np.linspace(np.pi/2, np.pi, sample_num)
        c_focal = calc_value_from_hue_1dlut(hue, focal_lut)
        chroma, lightness = calc_icn_xy_from_c_focal(
            cmap_lut_val_c=rr, degree_c=degree_data, c_focal=c_focal)

    return lightness.flatten(), chroma.flatten()


def calc_degree_from_cl_data_using_l_focal(cl_data, l_focal):
    """
    chroma-lightness のデータから degree を計算
    """
    chroma = cl_data[..., 0]
    lightness = cl_data[..., 1]

    # chroma == 0 は -np.pi/2 or np.pi/2 になる
    degree = np.where(
        chroma != 0,
        np.arctan((lightness - l_focal) / chroma),
        (np.pi / 2) * np.sign(lightness - l_focal)
    )

    return degree


def calc_degree_from_cl_data_using_c_focal(cl_data, c_focal):
    """
    chroma-lightness のデータから degree を計算
    """
    chroma = cl_data[..., 0]
    lightness = cl_data[..., 1]

    return np.arctan(lightness / (chroma - c_focal)) + np.pi


def _check_chroma_map_lut_interpolation(hue):
    """
    interpolate_chroma_map_lut() の動作確認用のデバッグコード。
    1. まずはLUT上の LC平面で確認
    2. 次に補間が働く LC平面で確認
    3. 今度は補間が働く ab平面で確認
    """
    # とりあえず L*C* 平面のポリゴン準備
    cl_inner = get_chroma_lightness_val_specfic_hue(hue, mcfl.BT709_BOUNDARY)
    cl_outer =\
        get_chroma_lightness_val_specfic_hue(hue, mcfl.BT2020_BOUNDARY)

    # cusp 準備
    lh_inner_lut = np.load(mcfl.BT709_BOUNDARY)
    lh_outer_lut = np.load(mcfl.BT2020_BOUNDARY)
    lcusp = mcfl.calc_l_cusp_specific_hue(hue, lh_inner_lut, lh_outer_lut)
    inner_cusp = mcfl.calc_cusp_in_lc_plane(hue, lh_inner_lut)
    outer_cusp = mcfl.calc_cusp_in_lc_plane(hue, lh_outer_lut)

    # l_cusp, l_focal, c_focal 準備
    l_cusp_lut = np.load(mcfl.L_CUSP_NAME)
    l_focal_lut = np.load(mcfl.L_FOCAL_NAME)
    c_focal_lut = np.load(mcfl.C_FOCAL_NAME)
    l_cusp = calc_value_from_hue_1dlut(hue, l_cusp_lut)
    l_focal = calc_value_from_hue_1dlut(hue, l_focal_lut)
    c_focal = calc_value_from_hue_1dlut(hue, c_focal_lut)

    # Chroma Mapping の Focalからの距離の LUT データ
    cmap_lut_l = np.load(CHROMA_MAP_LUT_LUMINANCE_NAME)
    cmap_lut_c = np.load(CHROMA_MAP_LUT_CHROMA_NAME)

    # st_degree, ed_degree を 1次元LUTの形で得る
    # st_degree_l[hue] = 30°, ed_degree_l[hue] = 120° 的な？
    st_degree_l, ed_degree_l, st_degree_c, ed_degree_c =\
        calc_chroma_map_degree(l_focal_lut, c_focal_lut)
    cmap_lut_h_sample = cmap_lut_l.shape[0]
    cmap_lut_d_sample = cmap_lut_l.shape[1]
    degree_l = linspace_2d(st_degree_l, ed_degree_l, cmap_lut_d_sample)
    degree_c = linspace_2d(st_degree_c, ed_degree_c, cmap_lut_d_sample)

    # とりあえず検証用のデータを準備
    # 一応、本番を想定して chroma-lightness から変換するように仕込む
    # hue-degree --> chroma-lightness --> hue_degree --> 補間的な？
    """ L_focal 基準データ """
    lightness_l, chroma_l = _make_debug_luminance_chroma_data_fixed_hue(
        hue=hue, hue_sample_num=cmap_lut_h_sample,
        st_degree_lut=st_degree_l, ed_degree_lut=ed_degree_l,
        focal_lut=l_focal_lut, focal_type="l")
    # lightness_l, chroma_l = _make_debug_luminance_chroma_data_fixed_hue2(
    #     hue=hue, focal_lut=l_focal_lut, focal_type="l")
    hue_array = np.ones(chroma_l.shape[0]) * hue
    cl_data_l = np.dstack((chroma_l, lightness_l))[0]
    test_degree_l = calc_degree_from_cl_data_using_l_focal(
        cl_data=cl_data_l,
        l_focal=calc_value_from_hue_1dlut(hue_array, l_focal_lut))
    test_hd_data_l = np.dstack((hue_array, test_degree_l))[0]

    """ C_focal 基準データ """
    lightness_c, chroma_c = _make_debug_luminance_chroma_data_fixed_hue(
        hue=hue, hue_sample_num=cmap_lut_h_sample,
        st_degree_lut=st_degree_c, ed_degree_lut=ed_degree_c,
        focal_lut=c_focal_lut, focal_type="c")
    # lightness_c, chroma_c = _make_debug_luminance_chroma_data_fixed_hue2(
    #     hue=hue, focal_lut=c_focal_lut, focal_type="c")
    hue_array = np.ones(chroma_l.shape[0]) * hue
    cl_data_c = np.dstack((chroma_c, lightness_c))[0]
    test_degree_c = calc_degree_from_cl_data_using_c_focal(
        cl_data=cl_data_c,
        c_focal=calc_value_from_hue_1dlut(hue_array, c_focal_lut))
    test_hd_data_c = np.dstack((hue_array, test_degree_c))[0]

    # まずは cmap_lut 値の Bilinear補間
    cmap_value_l = interpolate_chroma_map_lut(
        cmap_hd_lut=cmap_lut_l, degree_min=st_degree_l,
        degree_max=ed_degree_l, data_hd=test_hd_data_l)
    print(cmap_value_l)

    # 補間して得られた cmap 値から CL平面上における座標を取得
    # icn_x_l, icn_y_l = calc_icn_xy_from_l_focal(
    #     cmap_lut_val_l=cmap_lut_l[h_idx], degree_l=degree_l, l_focal=l_focal)
    # icn_x_c, icn_y_c = calc_icn_xy_from_c_focal(
    #     cmap_lut_val_c=cmap_lut_c[h_idx], degree_c=degree_c, c_focal=c_focal)

    # _debug_plot_check_lightness_mapping_specific_hue(
    #     hue, cl_inner, cl_outer, lcusp, inner_cusp, outer_cusp,
    #     l_cusp, l_focal, c_focal,
    #     x_val=chroma_l, y_val=lightness_l,
    #     focal_type="L_focal", idx=0)
    # _debug_plot_check_lightness_mapping_specific_hue(
    #     hue, cl_inner, cl_outer, lcusp, inner_cusp, outer_cusp,
    #     l_cusp, l_focal, c_focal,
    #     x_val=chroma_c, y_val=lightness_c,
    #     focal_type="C_focal", idx=0)


def calc_degree_min_max_for_interpolation():
    """
    cmapの補間用の degree_min, degree_max を計算する
    """
    l_focal_lut = np.load(mcfl.L_FOCAL_NAME)
    c_focal_lut = np.load(mcfl.C_FOCAL_NAME)
    st_degree_l, ed_degree_l, st_degree_c, ed_degree_c =\
        calc_chroma_map_degree(l_focal_lut, c_focal_lut)


def interpolate_chroma_map_lut(cmap_hd_lut, degree_min, degree_max, data_hd):
    """
    Chroma Mapping の LUT が任意の Input に
    対応できるように補間をする。

    LUTは Hue-Degree の2次元LUTである。
    これを Bilinear補間する。

    cmap_hd_lut: array_like
        Hue, Degree に対応する Chroma値が入っているLUT。

    degree_min: array_like
        cmap_hd_lut の 各 h_idx に対する degree の
        開始角度(degree_min)、終了角度(degree_max) が
        入っている。
        print(degree_min[h_idx]) ==> 0.4pi みたいなイメージ

    data_hd: array_like(shape is (N, 2))
        data_hd[..., 0]: Hue Value
        data_hd[..., 1]: Degree

    """
    # 補間に利用するLUTのIndexを算出
    hue_data = data_hd[..., 0]
    degree_data = data_hd[..., 1]
    hue_sample_num = get_hue_sample_num()
    hue_index_max = hue_sample_num - 1
    degree_sample_num = get_degree_sample_num()
    degree_index_max = degree_sample_num - 1

    # 1. h_idx
    h_idx_float = hue_data / (2 * np.pi) * (hue_index_max)
    h_idx_low = np.int16(h_idx_float)
    h_idx_high = h_idx_low + 1
    h_idx_low = np.clip(h_idx_low, 0, hue_index_max)
    h_idx_high = np.clip(h_idx_high, 0, hue_index_max)

    degree_lmin = degree_min[h_idx_low]
    degree_lmax = degree_max[h_idx_low]
    degree_hmin = degree_min[h_idx_high]
    degree_hmax = degree_max[h_idx_high]

    # 2. d_idx
    d_idx_l_float = (degree_data - degree_lmin)\
        / (degree_lmax - degree_lmin) * degree_index_max
    d_idx_l_float = np.clip(d_idx_l_float, 0, degree_index_max)

    d_idx_ll = np.int16(d_idx_l_float)
    d_idx_lh = d_idx_ll + 1
    d_idx_h_float = (degree_data - degree_hmin)\
        / (degree_hmax - degree_hmin) * degree_index_max
    d_idx_h_float = np.clip(d_idx_h_float, 0, degree_index_max)
    d_idx_hl = np.int16(d_idx_h_float)
    d_idx_hh = d_idx_hl + 1
    d_idx_ll = np.clip(d_idx_ll, 0, degree_index_max)
    d_idx_lh = np.clip(d_idx_lh, 0, degree_index_max)
    d_idx_hl = np.clip(d_idx_hl, 0, degree_index_max)
    d_idx_hh = np.clip(d_idx_hh, 0, degree_index_max)

    # 3. r_low, r_high
    r_low = d_idx_lh - d_idx_l_float
    r_high = d_idx_hh - d_idx_h_float

    # 4. interpolation in degree derection
    intp_d_low = r_low * cmap_hd_lut[h_idx_low, d_idx_ll]\
        + (1 - r_low) * cmap_hd_lut[h_idx_low, d_idx_lh]
    intp_d_high = r_high * cmap_hd_lut[h_idx_high, d_idx_hl]\
        + (1 - r_high) * cmap_hd_lut[h_idx_high, d_idx_hh]

    # 6. final_r
    final_r = h_idx_high - h_idx_float

    # 7. interpolation in hue direction
    intp_data = final_r * intp_d_low + (1 - final_r) + intp_d_high

    return intp_data


def main_func():
    # これが めいんるーちん
    # make_chroma_map_lut()

    # こっから先は えくすぺりめんたるな るーちん
    # とりあえず、任意の Hue において一発 Lightness Mapping してみる
    # _try_lightness_mapping_specific_hue(hue=00/360*2*np.pi)
    # _try_lightness_mapping_specific_hue(hue=90/360*2*np.pi)
    # _try_lightness_mapping_specific_hue(hue=180/360*2*np.pi)
    # _try_lightness_mapping_specific_hue(hue=270/360*2*np.pi)
    # _check_calc_cmap_on_lc_plane(hue=00/360*2*np.pi)
    # _check_calc_cmap_on_lc_plane(hue=90/360*2*np.pi)
    # _check_calc_cmap_on_lc_plane(hue=180/360*2*np.pi)
    # _check_calc_cmap_on_lc_plane(hue=270/360*2*np.pi)
    # _check_chroma_map_lut_data(30)
    # _check_chroma_map_lut_data(100)
    _check_chroma_map_lut_interpolation(hue=30/360*2*np.pi)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
