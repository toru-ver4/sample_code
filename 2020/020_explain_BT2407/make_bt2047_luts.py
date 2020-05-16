# -*- coding: utf-8 -*-
"""
BT2407 実装用の各種LUTを作成する
===============================

"""

# import standard libraries
import os
import ctypes
import time

# import third-party libraries
from sympy import symbols
import numpy as np
from multiprocessing import Pool, cpu_count, Array
from scipy import signal, interpolate
import matplotlib.pyplot as plt


# import my libraries
import cielab as cl
import color_space as cs
from bt2407_parameters import L_SAMPLE_NUM_MAX, H_SAMPLE_NUM_MAX,\
    GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE, GAMUT_BOUNDARY_LUT_HUE_SAMPLE,\
    get_gamut_boundary_lut_name, get_l_cusp_name, get_focal_name,\
    DIPS_150_SAMPLE_ST_BT2020, DIPS_150_SAMPLE_ED_BT2020,\
    DIPS_300_SAMPLE_ST_BT2020, DIPS_300_SAMPLE_ED_BT2020,\
    DIPS_150_SAMPLE_ST_P3, DIPS_150_SAMPLE_ED_P3,\
    DIPS_300_SAMPLE_ST_P3, DIPS_300_SAMPLE_ED_P3,\
    L_FOCAL_240_INDEX_BT2020, L_FOCAL_240_INDEX_P3,\
    C_FOCAL_MAX_VALUE, LPF_WN_PARAM, LPF_NN_PARAM,\
    CHROMA_MAP_DEGREE_SAMPLE_NUM, get_chroma_map_lut_name
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


shared_array = Array(
    typecode_or_type=ctypes.c_float,
    size_or_initializer=L_SAMPLE_NUM_MAX*H_SAMPLE_NUM_MAX)

shared_array2 = Array(
    typecode_or_type=ctypes.c_float,
    size_or_initializer=L_SAMPLE_NUM_MAX*H_SAMPLE_NUM_MAX)


def load_cusp_focal_lut(
        outer_color_space_name=cs.BT2020,
        inner_color_space_name=cs.BT709):
    # l_cusp, l_focal, c_focal 準備
    l_cusp_lut = np.load(
        get_l_cusp_name(
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name))
    l_focal_lut = np.load(
        get_focal_name(
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name,
            focal_type="Lfocal"))
    c_focal_lut = np.load(
        get_focal_name(
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name,
            focal_type="Cfocal"))

    return l_cusp_lut, l_focal_lut, c_focal_lut


def calc_cusp_focal_specific_hue(
        hue=np.deg2rad(30),
        outer_color_space_name=cs.BT2020,
        inner_color_space_name=cs.BT709):

    l_cusp_lut, l_focal_lut, c_focal_lut = load_cusp_focal_lut(
        outer_color_space_name=outer_color_space_name,
        inner_color_space_name=inner_color_space_name)
    l_cusp = calc_value_from_hue_1dlut(hue, l_cusp_lut)
    l_focal = calc_value_from_hue_1dlut(hue, l_focal_lut)
    c_focal = calc_value_from_hue_1dlut(hue, c_focal_lut)

    return l_cusp, l_focal, c_focal


def solve_chroma_wrapper(args):
    chroma = cl.solve_chroma(**args)
    s_idx = args['h_sample_num'] * args['l_idx'] + args['h_idx']
    shared_array[s_idx] = chroma


def solve_chroma_wrapper_fast(args):
    chroma = cl.solve_chroma_fast(**args)
    s_idx = args['h_sample_num'] * args['l_idx'] + args['h_idx']
    shared_array[s_idx] = chroma


def solve_chroma_wrapper_fastest(args):
    chroma = cl.solve_chroma_fastest(**args)
    s_idx = args['h_sample_num'] * args['l_idx']
    shared_array[s_idx:s_idx+args['h_sample_num']] = chroma


def make_chroma_array(primaries=cs.get_primaries(cs.BT709),
                      l_sample_num=L_SAMPLE_NUM_MAX,
                      h_sample_num=H_SAMPLE_NUM_MAX):
    """
    L*a*b* 空間における a*b*平面の境界線プロットのために、
    各L* における 境界線の Chroma を計算する。
    """

    l, c, h = symbols('l, c, h')
    rgb_exprs = cl.lab_to_rgb_expr(l, c, h, primaries=primaries)
    l_vals = np.linspace(0, 100, l_sample_num)
    h_vals = np.linspace(0, 2*np.pi, h_sample_num)
    for l_idx, l_val in enumerate(l_vals):
        args = []
        for h_idx, h_val in enumerate(h_vals):
            d = dict(
                l_val=l_val, l_idx=l_idx, h_val=h_val, h_idx=h_idx,
                rgb_exprs=rgb_exprs, l=l, c=c, h=h,
                l_sample_num=l_sample_num, h_sample_num=h_sample_num)
            args.append(d)
        with Pool(cpu_count()) as pool:
            pool.map(solve_chroma_wrapper, args)

    chroma = np.array(
        shared_array[:l_sample_num * h_sample_num]).reshape(
            (l_sample_num, h_sample_num))
    return chroma


def make_chroma_array_fast(
        color_space_name=cs.BT709,
        l_sample_num=L_SAMPLE_NUM_MAX,
        h_sample_num=H_SAMPLE_NUM_MAX):
    """
    L*a*b* 空間における a*b*平面の境界線プロットのために、
    各L* における 境界線の Chroma を計算する。
    高速版。
    """

    l_vals = np.linspace(0, 100, l_sample_num)
    h_vals = np.linspace(0, 2*np.pi, h_sample_num)
    for l_idx, l_val in enumerate(l_vals):
        args = []
        for h_idx, h_val in enumerate(h_vals):
            d = dict(
                l_val=l_val, l_idx=l_idx, h_val=h_val, h_idx=h_idx,
                l_sample_num=l_sample_num, h_sample_num=h_sample_num,
                color_space_name=color_space_name)
            args.append(d)
        with Pool(cpu_count()) as pool:
            pool.map(solve_chroma_wrapper_fast, args)

    chroma = np.array(
        shared_array[:l_sample_num * h_sample_num]).reshape(
            (l_sample_num, h_sample_num))
    return chroma


def make_chroma_array_fastest(
        color_space_name=cs.BT709,
        l_sample_num=L_SAMPLE_NUM_MAX,
        h_sample_num=H_SAMPLE_NUM_MAX):
    """
    L*a*b* 空間における a*b*平面の境界線プロットのために、
    各L* における 境界線の Chroma を計算する。
    高速版。
    """
    l_vals = np.linspace(0, 100, l_sample_num)
    h_vals = np.linspace(0, 2*np.pi, h_sample_num)
    args = []
    for l_idx, l_val in enumerate(l_vals):
        d = dict(
            l_val=l_val, l_idx=l_idx, h_vals=h_vals,
            l_sample_num=l_sample_num, h_sample_num=h_sample_num,
            color_space_name=color_space_name)
        args.append(d)
    with Pool(cpu_count()) as pool:
        pool.map(solve_chroma_wrapper_fastest, args)

    chroma = np.array(
        shared_array[:l_sample_num * h_sample_num]).reshape(
            (l_sample_num, h_sample_num))
    return chroma


def make_gamut_bondary_lut(
        l_sample_num=GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE,
        h_sample_num=GAMUT_BOUNDARY_LUT_HUE_SAMPLE,
        color_space_name=cs.BT709):
    chroma = make_chroma_array(
        primaries=cs.get_primaries(color_space_name),
        l_sample_num=l_sample_num, h_sample_num=h_sample_num)
    fname = get_gamut_boundary_lut_name(
        color_space_name, l_sample_num, h_sample_num)
    np.save(fname, chroma)


def make_gamut_bondary_lut_fast(
        l_sample_num=GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE,
        h_sample_num=GAMUT_BOUNDARY_LUT_HUE_SAMPLE,
        color_space_name=cs.BT709):
    chroma = make_chroma_array_fast(
        color_space_name=color_space_name,
        l_sample_num=l_sample_num, h_sample_num=h_sample_num)
    fname = get_gamut_boundary_lut_name(
        color_space_name, l_sample_num, h_sample_num)
    np.save(fname, chroma)


def make_gamut_bondary_lut_fastest(
        l_sample_num=GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE,
        h_sample_num=GAMUT_BOUNDARY_LUT_HUE_SAMPLE,
        color_space_name=cs.BT709):
    chroma = make_chroma_array_fastest(
        color_space_name=color_space_name,
        l_sample_num=l_sample_num, h_sample_num=h_sample_num)
    fname = get_gamut_boundary_lut_name(
        color_space_name, l_sample_num, h_sample_num)
    np.save(fname, chroma)


def make_gamut_boundary_lut_all():
    # L*a*b* 全体のデータを算出
    start = time.time()
    make_gamut_bondary_lut(color_space_name=cs.BT709)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    start = time.time()
    make_gamut_bondary_lut(color_space_name=cs.BT2020)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    start = time.time()
    make_gamut_bondary_lut(color_space_name=cs.P3_D65)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")


def make_gamut_boundary_lut_all_fast():
    # L*a*b* 全体のデータを算出
    start = time.time()
    make_gamut_bondary_lut_fast(color_space_name=cs.BT709)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    start = time.time()
    make_gamut_bondary_lut_fast(color_space_name=cs.BT2020)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    start = time.time()
    make_gamut_bondary_lut_fast(color_space_name=cs.P3_D65)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")


def make_gamut_boundary_lut_all_fastest():
    # L*a*b* 全体のデータを算出
    start = time.time()
    make_gamut_bondary_lut_fastest(color_space_name=cs.BT709)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    start = time.time()
    make_gamut_bondary_lut_fastest(color_space_name=cs.BT2020)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    start = time.time()
    make_gamut_bondary_lut_fastest(color_space_name=cs.P3_D65)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")


def calc_intercsection_with_lightness_axis(inter_cusp, outer_cusp):
    """
    calculate the intersection of the two cusps
    and lightness axis in L*-Chroma plane.

    Returns
    -------
    touple
        (L*star, Chroma). It is the coordinate of the L_cusp.
    """
    x1 = inter_cusp[1]
    y1 = inter_cusp[0]
    x2 = outer_cusp[1]
    y2 = outer_cusp[0]

    y = y2 - (y2 - y1) / (x2 - x1) * x2

    return (y, 0)


def calc_cusp_in_lc_plane(hue, lh_lut):
    """
    calculate Cusp in a specific L*-C* plane.

    Parameters
    ----------
    hue : float
        hue(the unit is radian)
    lh_lut : array_like (2D)
        L*-Chroma 2D-LUT.

    Returns
    -------
    touple
        (L*star, Chroma). It is the coordinate of the Cusp.

    """
    l_sample = np.linspace(0, 100, GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE)
    h_sample = np.ones_like(l_sample) * hue
    lh_sample = np.dstack((l_sample, h_sample))
    chroma_for_each_l = cl.bilinear_interpolation(lh=lh_sample, lut2d=lh_lut)
    cusp_idx = np.argmax(chroma_for_each_l)

    return np.array((l_sample[cusp_idx], chroma_for_each_l[cusp_idx]))


def calc_l_cusp_specific_hue(hue, inner_lut, outer_lut):
    """
    Parameters
    ----------
    hue : float
        hue(the unit is radian)
    inner_lut : array_like (2D)
        L*-Chroma 2D-LUT for inner gamut.
    outer_lut : array_like (2D)
        L*-Chroma 2D-LUT for outer gamut.
    """
    inner_cusp = calc_cusp_in_lc_plane(hue, inner_lut)
    outer_cusp = calc_cusp_in_lc_plane(hue, outer_lut)

    lcusp = calc_intercsection_with_lightness_axis(inner_cusp, outer_cusp)

    return lcusp[0]


def calc_l_cusp(
        inner_color_space_name=cs.BT709, outer_color_space_name=cs.BT2020):
    inner_lut = np.load(
        get_gamut_boundary_lut_name(inner_color_space_name))
    outer_lut = np.load(
        get_gamut_boundary_lut_name(outer_color_space_name))
    l_cusp = []
    h_sample = GAMUT_BOUNDARY_LUT_HUE_SAMPLE
    hue_list = np.linspace(0, 2*np.pi, h_sample)

    for hue in hue_list:
        lll = calc_l_cusp_specific_hue(hue, inner_lut, outer_lut)
        l_cusp.append(lll)
    l_cusp = np.array(l_cusp)

    return l_cusp


def low_pass_filter2(x, nn=4, wn=0.25):
    b1, a1 = signal.bessel(nn, wn, "low")
    result = signal.filtfilt(b1, a1, x)

    return result


def get_dips_value_around_135(l_cusp, outer_color_space_name=cs.BT2020):
    """
    135°付近の凹みの L* 値および、それを指す Hue の Index を計算する。
    """
    if outer_color_space_name == cs.BT2020:
        dips_150 = np.min(
            l_cusp[DIPS_150_SAMPLE_ST_BT2020:DIPS_150_SAMPLE_ED_BT2020])
        dips_150_idx = np.argmin(
            l_cusp[DIPS_150_SAMPLE_ST_BT2020:DIPS_150_SAMPLE_ED_BT2020])
    else:
        dips_150 = np.min(
            l_cusp[DIPS_150_SAMPLE_ST_P3:DIPS_150_SAMPLE_ED_P3])
        dips_150_idx = np.argmin(
            l_cusp[DIPS_150_SAMPLE_ST_P3:DIPS_150_SAMPLE_ED_P3])

    return dips_150, dips_150_idx


def get_dips_value_around_300(l_cusp, outer_color_space_name=cs.BT2020):
    """
    300°付近の凹みの L* 値および、それを指す Hue の Index を計算する。
    """
    if outer_color_space_name == cs.BT2020:
        dips_300 = np.min(
            l_cusp[DIPS_300_SAMPLE_ST_BT2020:DIPS_300_SAMPLE_ED_BT2020])
        dips_300_idx = np.argmin(
            l_cusp[DIPS_300_SAMPLE_ST_BT2020:DIPS_300_SAMPLE_ED_BT2020])
        dips_300_idx += DIPS_300_SAMPLE_ST_BT2020
    else:
        dips_300 = np.min(
            l_cusp[DIPS_300_SAMPLE_ST_P3:DIPS_300_SAMPLE_ED_P3])
        dips_300_idx = np.argmin(
            l_cusp[DIPS_300_SAMPLE_ST_P3:DIPS_300_SAMPLE_ED_P3])
        dips_300_idx += DIPS_300_SAMPLE_ST_P3

    return dips_300, dips_300_idx


def calc_l_focal(l_cusp, outer_color_space_name=cs.BT2020):
    """
    l_cusp に修正を加えた l_focal を求める

    1. min, max の設定。np.clip() で dips300, dips150 の範囲内に制限
    2. 240°～300° にかけて緩やかにスロープさせる
    """
    if outer_color_space_name == cs.BT2020:
        l_focal_240_index = L_FOCAL_240_INDEX_BT2020
    else:
        l_focal_240_index = L_FOCAL_240_INDEX_P3
    dips_150, _ = get_dips_value_around_135(
        l_cusp, outer_color_space_name=outer_color_space_name)
    dips_300, dips_300_idx = get_dips_value_around_300(
        l_cusp, outer_color_space_name=outer_color_space_name)
    decrement_sample = dips_300_idx - l_focal_240_index + 1
    decrement_data = np.linspace(dips_150, dips_300, decrement_sample)
    l_cusp_low_pass = low_pass_filter2(
        l_cusp, nn=LPF_NN_PARAM, wn=LPF_WN_PARAM)
    l_focal = np.clip(l_cusp_low_pass, dips_300, dips_150)
    l_focal[l_focal_240_index:dips_300_idx + 1] = decrement_data
    l_focal[dips_300_idx:] = dips_300

    _debug_plot_l_cusp(
        l_cusp, l_focal, dips_150, dips_300, l_cusp_low_pass,
        outer_color_space_name)

    return l_focal


def calc_intersection_with_chroma_axis(inner_cusp, outer_cusp):
    """
    calculate the intersection of the two cusps
    and chroma axis in L*-Chroma plane.

    Returns
    -------
    touple
        (L*star, Chroma). It is the coordinate of the L_cusp.
    """
    x1 = inner_cusp[1]
    y1 = inner_cusp[0]
    x2 = outer_cusp[1]
    y2 = outer_cusp[0]

    div_val = (y2 - y1)

    x = x2 - (x2 - x1) / div_val * y2 if div_val != 0 else 0

    return (0, x)


def calc_c_focal_specific_hue(hue, inner_lut, outer_lut):
    """
    Parameters
    ----------
    hue : float
        hue(the unit is radian)
    inner_lut : array_like (2D)
        L*-Chroma 2D-LUT for inner gamut.
    outer_lut : array_like (2D)
        L*-Chroma 2D-LUT for outer gamut.
    """
    inner_cusp = calc_cusp_in_lc_plane(hue, inner_lut)
    outer_cusp = calc_cusp_in_lc_plane(hue, outer_lut)

    c_focal = calc_intersection_with_chroma_axis(inner_cusp, outer_cusp)

    return c_focal[1]


def interpolate_where_value_is_zero(x, y):
    """
    ゼロ割の影響で y=0 となってしまっている箇所がある。
    線形補間でそこを別の値に置き換える。
    """
    not_zero_idx = (y != 0)
    f = interpolate.interp1d(x[not_zero_idx], y[not_zero_idx])
    y_new = f(x)

    return y_new


def calc_c_focal(
        outer_color_space_name=cs.BT2020,
        inner_color_space_name=cs.BT709):
    inner_lut = np.load(
        get_gamut_boundary_lut_name(inner_color_space_name))
    outer_lut = np.load(
        get_gamut_boundary_lut_name(outer_color_space_name))
    c_focal = []
    h_sample = GAMUT_BOUNDARY_LUT_HUE_SAMPLE
    hue_list = np.linspace(0, 2*np.pi, h_sample)

    for idx, hue in enumerate(hue_list):
        lll = calc_c_focal_specific_hue(hue, inner_lut, outer_lut)
        c_focal.append(lll)
        # break
    c_focal = np.abs(np.array(c_focal))

    # 色々と問題があるので補間とかLPFとか処理する
    c_focal_interp = interpolate_where_value_is_zero(hue_list, c_focal)
    c_focal_interp[c_focal_interp > C_FOCAL_MAX_VALUE] = C_FOCAL_MAX_VALUE
    c_focal_lpf = low_pass_filter2(
        c_focal_interp, nn=LPF_NN_PARAM, wn=LPF_WN_PARAM)
    _debug_plot_c_focal(c_focal, c_focal_lpf, outer_color_space_name)

    return c_focal_lpf


def make_focal_lut(
        outer_color_space_name=cs.BT2020,
        inner_color_space_name=cs.BT709):
    # L Cusp
    l_cusp = calc_l_cusp(
        outer_color_space_name=outer_color_space_name,
        inner_color_space_name=inner_color_space_name)
    np.save(
        get_l_cusp_name(
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name),
        l_cusp)

    # L_focal
    l_focal = calc_l_focal(
        l_cusp, outer_color_space_name=outer_color_space_name)
    np.save(
        get_focal_name(
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name,
            focal_type="Lfocal"),
        l_focal)

    # C_focal
    c_focal = calc_c_focal(
        outer_color_space_name=outer_color_space_name,
        inner_color_space_name=inner_color_space_name)
    np.save(
        get_focal_name(
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name,
            focal_type="Cfocal"),
        c_focal)


def _debug_plot_l_cusp(
        l_cusp, l_focal, dips_150, dips_300, low_pass, outer_color_space_name):

    x = np.linspace(0, 360, len(l_cusp))
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title=f"L focal {outer_color_space_name} to ITU-R BT.709",
        graph_title_size=None,
        xlabel="Hue",
        ylabel="Lightness",
        axis_label_size=None,
        legend_size=17,
        xlim=[-10, 370],
        ylim=[30, 103],
        xtick=[x * 45 for x in range(9)],
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(x, l_cusp, lw=5, label="L_cusp")
    ax1.plot(x, low_pass, lw=3, c="#B0B0B0", label="Low Pass")
    ax1.plot(x, np.ones_like(x) * dips_150, 'k--', label=f"L*={dips_150:.2f}",
             alpha=0.5)
    ax1.plot(x, np.ones_like(x) * dips_300, 'k:', label=f"L*={dips_300:.2f}",
             alpha=0.5)
    ax1.plot(x, l_focal, label="L_focal")
    # ax1.plot(l_cusp, label="Original")
    # ax1.plot(low_pass, label="Low Pass")
    plt.legend(loc='lower center')
    plt.savefig(
        f"./figures/L_focal_outer_gamut_{outer_color_space_name}.png",
        bbox_inches='tight', pad_inches=0.1)
    # plt.show()


def _debug_plot_c_focal(c_focal, low_pass, outer_color_space_name):
    x = np.linspace(0, 360, len(c_focal))
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title=f"C focal {outer_color_space_name} to ITU-R BT.709",
        graph_title_size=None,
        xlabel="Hue",
        ylabel="Chroma",
        axis_label_size=None,
        legend_size=17,
        xlim=[-10, 370],
        ylim=[0, C_FOCAL_MAX_VALUE * 1.2],
        xtick=[x * 45 for x in range(9)],
        ytick=None,
        linewidth=3)
    # オリジナルをプロット
    ax1.plot(x, c_focal, ':', c="#808080", lw=5, label="C_focal_raw")
    zero_idx = (c_focal == 0)
    ax1.plot(x[zero_idx], c_focal[zero_idx],
             'x', ms=10, mew=5, label="Zero Division Error")

    # ゼロ割の欠損値を線形補間
    c_focal_interp = interpolate_where_value_is_zero(x, c_focal)
    ax1.plot(x, c_focal_interp, '-', c="#808080",
             lw=5, label="C_focal_interpolated")

    ax1.plot(x, low_pass, '-', lw=3, label="Apply LPF")
    plt.legend(loc='upper right')
    plt.savefig(
        f"./figures/C_focal_outer_gamut_{outer_color_space_name}.png",
        bbox_inches='tight', pad_inches=0.1)
    # plt.show()


def get_chroma_lightness_val_specfic_hue(
        hue=30/360*2*np.pi,
        lh_lut_name=get_gamut_boundary_lut_name(cs.BT709)):
    lh_lut = np.load(lh_lut_name)
    lstar = np.linspace(0, 100, lh_lut.shape[0])
    hue_list = np.ones((lh_lut.shape[1])) * hue
    lh = np.dstack([lstar, hue_list])
    chroma = cl.bilinear_interpolation(lh, lh_lut)

    return np.dstack((chroma, lstar))[0]


def calc_value_from_hue_1dlut(val, lut):
    """
    Lfocal や Cfocal など Hue値が入力となっている
    1DLUTの補間計算をして出力する。
    """
    x = np.linspace(0, 2*np.pi, len(lut))
    f = interpolate.interp1d(x, lut)
    y = f(val)

    return y


def calc_chroma_map_degree2(l_focal, c_focal, inner_cusp_lc):
    """
    Chroma Mapping の Destination の位置用の LUT の
    Start, End の Degree を計算。
    当初の設計だと特定条件で誤差が生じたため、ver2 を別途作成した。

    ver2 は c_focal 側の st_degree が l_focal - c_focal の直線
    ではなく、inner_cusp の座標となっている。

    少し多めに確保しておかないと補間計算での誤差が生じるため。
    """
    st_degree_l = -np.arctan(l_focal/c_focal)
    ed_degree_l = np.pi/2 * np.ones_like(st_degree_l)
    angle_inner_cusp = -np.arctan(
        inner_cusp_lc[..., 0] / (c_focal - inner_cusp_lc[..., 1]))
    st_degree_c = np.pi + (angle_inner_cusp * 0.95) + (st_degree_l * 0.05)
    ed_degree_c = np.pi * np.ones_like(st_degree_c)

    return st_degree_l, ed_degree_l, st_degree_c, ed_degree_c


def _calc_ab_coef_from_cl_point(cl_point):
    """
    Chroma-Luminance のデータ1点1点に対して、
    隣接し合う2点を結んでできる直線の
    y=ax+b の a, b を求める。

    focal を基準とした直線と Gamut Boundary の
    交点の算出で使用する。
    """
    x_list = cl_point[..., 0]
    y_list = cl_point[..., 1]

    a = (y_list[1:] - y_list[:-1]) / (x_list[1:] - x_list[:-1])
    b = y_list[1:] - a * x_list[1:]

    return a, b


def solve_equation_for_intersection(
        cl_point, a1, b1, a2, b2, focal="L_Focal", inner_cusp=None):
    """
    Focal へ向かう・収束する直線と Inner Gamut Boundary の交点を求める。
    この交点が Out of Gamut の Mapping先の値となる。

    Parameters
    ----------
    cl_point : array_like (2d array)
        Inner Gamut Boundary の Chroma-Lightness データ。
        交点の方程式を解いた後で解が適切かどうかの判断で使用する。
    a1 : array_like
        y=ax+b の a のパラメータ。
        Focal を中心とした直線。
    b1 : array_like
        y=ax+b の b のパラメータ。
        Focal を中心とした直線。
    a2 : array_like
        y=ax+b の a のパラメータ。
        Inner Gamut Boundary の隣接するサンプルを結んだ直線
    b2 : array_like
        y=ax+b の b のパラメータ。
        Inner Gamut Boundary の隣接するサンプルを結んだ直線
    focal : str
        focal の種類を設定。C_Focal は特別処理が必要なので
        その分岐用。
    inner_cusp : array_like
        C_Focal の特別処理用。
        C_Focal の交点の方程式の解の Lightness値は、
        Inner Gamut の Cusp よりも小さい必要がある。
        その判別式で使用する。
    hue : float
        hue. unit is radian. for debug plot.
    """
    # 1次元方程式っぽいのを解いて交点を算出
    icn_x = (b1[:, np.newaxis] - b2[np.newaxis, :])\
        / (a2[np.newaxis, :] - a1[:, np.newaxis])
    icn_y = a1[:, np.newaxis] * icn_x + b1[:, np.newaxis]

    # 交点から有効点？を抽出
    # print(icn_x.shape)
    ok_src_idx_x_0 = (icn_x >= cl_point[:-1, 0]) & (icn_x <= cl_point[1:, 0])
    ok_src_idx_x_1 = (icn_x <= cl_point[:-1, 0]) & (icn_x >= cl_point[1:, 0])
    ok_src_idx_x = ok_src_idx_x_0 | ok_src_idx_x_1
    if focal == "L_Focal":
        ok_src_idx_y = (icn_y >= cl_point[:-1, 1]) & (icn_y <= cl_point[1:, 1])
    else:
        ok_src_idx_y0\
            = (icn_y >= cl_point[:-1, 1]) & (icn_y <= cl_point[1:, 1])
        ok_src_idx_y1 = (icn_y < inner_cusp)
        ok_src_idx_y = ok_src_idx_y0 & ok_src_idx_y1
    ok_src_idx = ok_src_idx_x & ok_src_idx_y
    ok_dst_idx = np.any(ok_src_idx, axis=-1)
    icn_valid_x = np.zeros((icn_x.shape[0]))
    icn_valid_y = np.zeros((icn_y.shape[0]))
    icn_valid_x[ok_dst_idx] = icn_x[ok_src_idx]
    icn_valid_y[ok_dst_idx] = icn_y[ok_src_idx]

    return icn_valid_x, icn_valid_y


def calc_distance_from_l_focal(chroma, lightness, l_focal):
    """
    L_Focal から 引数で指定した Chroma-Lightness までの距離を求める。
    """
    distance = ((chroma) ** 2 + (lightness - l_focal) ** 2) ** 0.5
    return distance


def calc_distance_from_c_focal(chroma, lightness, c_focal):
    """
    C_Focal から 引数で指定した Chroma-Lightness までの距離を求める。
    """
    distance = ((chroma - c_focal) ** 2 + (lightness) ** 2) ** 0.5
    return distance


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

    graph_name = f"./cl_plane_seq/cmap_lut_{focal_type}_{idx:04d}.png"
    plt.legend(loc='upper right')
    # plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(graph_name)  # オプション付けるとエラーになるので外した
    # plt.show()


def make_chroma_map_lut_specific_hue(
        hue=30/360*2*np.pi, idx=0,
        inner_color_space_name=cs.BT709,
        outer_color_space_name=cs.BT2020):
    """
    Lfocal, Cfocal を中心とする放射線状のデータが
    どの Chroma値にマッピングされるかを示すLUTを作る。
    """
    print(f"hue={np.rad2deg(hue):.2f}")
    # maping 先となる BT.709 の Gamut Boundary データを作成
    cl_inner = get_chroma_lightness_val_specfic_hue(
        hue, get_gamut_boundary_lut_name(inner_color_space_name))

    # 境界値の計算に使用する Cusp を作成
    lh_inner_lut = np.load(
        get_gamut_boundary_lut_name(inner_color_space_name))
    inner_cusp = calc_cusp_in_lc_plane(hue, lh_inner_lut)

    # l_cusp, l_focal, c_focal 準備
    l_focal_lut = np.load(
        get_focal_name(
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name,
            focal_type="Lfocal"))
    c_focal_lut = np.load(
        get_focal_name(
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name,
            focal_type="Cfocal"))
    l_focal = calc_value_from_hue_1dlut(hue, l_focal_lut)
    c_focal = calc_value_from_hue_1dlut(hue, c_focal_lut)

    st_degree_l, ed_degree_l, st_degree_c, ed_degree_c =\
        calc_chroma_map_degree2(l_focal, c_focal, inner_cusp)

    # Lfocal用のサンプル点作成
    degree = np.linspace(
        st_degree_l, ed_degree_l, CHROMA_MAP_DEGREE_SAMPLE_NUM)
    a1_l = np.tan(degree)
    b1_l = l_focal * np.ones_like(degree)

    # Cfocal用のサンプル点作成
    degree = np.linspace(
        st_degree_c, ed_degree_c, CHROMA_MAP_DEGREE_SAMPLE_NUM)
    a1_c = np.tan(degree)
    b1_c = -a1_c * c_focal

    # 各 cl_point の2点間の直線 y=ax+b の a, b の値を出す
    # cl_inner = cl_inner[::48]
    a2, b2 = _calc_ab_coef_from_cl_point(cl_inner)

    # GamutBoundaryの直線群とFlcalの直線群の交点を求める。(L_focal)
    icn_x_l, icn_y_l = solve_equation_for_intersection(
        cl_inner, a1_l, b1_l, a2, b2)

    # GamutBoundaryの直線群とFlcalの直線群の交点を求める。(C_focal)
    icn_x_c, icn_y_c = solve_equation_for_intersection(
        cl_inner, a1_c, b1_c, a2, b2, focal="C_Focal",
        inner_cusp=inner_cusp[0])

    # cl_outer = get_chroma_lightness_val_specfic_hue(
    #     hue, get_gamut_boundary_lut_name(outer_color_space_name))
    # lh_outer_lut = np.load(
    #     get_gamut_boundary_lut_name(outer_color_space_name))
    # outer_cusp = calc_cusp_in_lc_plane(hue, lh_outer_lut)
    # l_cusp_lut = np.load(
    #     get_l_cusp_name(
    #         outer_color_space_name=outer_color_space_name,
    #         inner_color_space_name=inner_color_space_name))
    # l_cusp = calc_value_from_hue_1dlut(hue, l_cusp_lut)
    # _debug_plot_chroma_map_lut_specific_hue(
    #     hue, cl_inner, cl_outer, l_cusp, inner_cusp, outer_cusp,
    #     l_cusp, l_focal, c_focal, icn_x_l, icn_y_l, focal_type="L_focal",
    #     idx=idx)
    # _debug_plot_chroma_map_lut_specific_hue(
    #     hue, cl_inner, cl_outer, l_cusp, inner_cusp, outer_cusp,
    #     l_cusp, l_focal, c_focal, icn_x_c, icn_y_c, focal_type="C_focal",
    #     idx=idx)

    cmap_l = calc_distance_from_l_focal(icn_x_l, icn_y_l, l_focal)
    cmap_c = calc_distance_from_c_focal(icn_x_c, icn_y_c, c_focal)

    return cmap_l, cmap_c


def thread_wrapper_make_chroma_map_lut(args):
    cmap_l, cmap_c = make_chroma_map_lut_specific_hue(**args)

    s_idx = GAMUT_BOUNDARY_LUT_HUE_SAMPLE * args['idx']
    shared_array[s_idx:s_idx+CHROMA_MAP_DEGREE_SAMPLE_NUM] = cmap_l
    shared_array2[s_idx:s_idx+GAMUT_BOUNDARY_LUT_HUE_SAMPLE] = cmap_c


def make_chroma_map_lut(
        outer_color_space_name=cs.BT2020,
        inner_color_space_name=cs.BT709):
    """
    Lfocal, Cfocal を中心とする放射線状のデータが
    どの Chroma値にマッピングされるかを示すLUTを作る。
    """
    hue_sample = GAMUT_BOUNDARY_LUT_HUE_SAMPLE
    hue_list = np.linspace(0, 2 * np.pi, hue_sample)
    args = []
    # cmap_l_buf = []
    # cmap_c_buf = []
    for idx, hue in enumerate(hue_list):
        # cmap_l, cmap_c = make_chroma_map_lut_specific_hue(
        #     hue=hue, idx=idx,
        #     inner_color_space_name=inner_color_space_name,
        #     outer_color_space_name=outer_color_space_name)
        # cmap_l_buf.append(cmap_l)
        # cmap_c_buf.append(cmap_c)
        args.append(
            dict(
                hue=hue, idx=idx,
                inner_color_space_name=inner_color_space_name,
                outer_color_space_name=outer_color_space_name)
        )
    with Pool(cpu_count()) as pool:
        pool.map(thread_wrapper_make_chroma_map_lut, args)

    cmap_l_lut = np.array(
        shared_array[:GAMUT_BOUNDARY_LUT_HUE_SAMPLE*CHROMA_MAP_DEGREE_SAMPLE_NUM]).reshape(
        (GAMUT_BOUNDARY_LUT_HUE_SAMPLE, CHROMA_MAP_DEGREE_SAMPLE_NUM))
    cmap_c_lut = np.array(
        shared_array2[:GAMUT_BOUNDARY_LUT_HUE_SAMPLE*CHROMA_MAP_DEGREE_SAMPLE_NUM]).reshape(
        (GAMUT_BOUNDARY_LUT_HUE_SAMPLE, CHROMA_MAP_DEGREE_SAMPLE_NUM))

    # 整形して .npy で保存
    # cmap_l_lut = np.array(cmap_l_buf)
    # cmap_c_lut = np.array(cmap_c_buf)

    np.save(
        get_chroma_map_lut_name(
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name,
            focal_type="Lfocal"),
        cmap_l_lut)
    np.save(
        get_chroma_map_lut_name(
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name,
            focal_type="Cfocal"),
        cmap_c_lut)


def make_chroma_mapping_lut(
        outer_color_space_name=cs.BT2020,
        inner_color_space_name=cs.BT709):
    pass


def main_func():
    # make_gamut_boundary_lut_all()
    # make_gamut_boundary_lut_all_fast()
    make_gamut_boundary_lut_all_fastest()
    # make_focal_lut(
    #     outer_color_space_name=cs.BT2020,
    #     inner_color_space_name=cs.BT709)
    # make_focal_lut(
    #     outer_color_space_name=cs.P3_D65,
    #     inner_color_space_name=cs.BT709)
    # make_chroma_map_lut(
    #     outer_color_space_name=cs.BT2020,
    #     inner_color_space_name=cs.BT709)
    # make_chroma_map_lut(
    #     outer_color_space_name=cs.P3_D65,
    #     inner_color_space_name=cs.BT709)
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
