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
    C_FOCAL_MAX_VALUE, LPF_WN_PARAM, LPF_NN_PARAM
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


def solve_chroma_wrapper(args):
    chroma = cl.solve_chroma(**args)
    s_idx = args['h_sample_num'] * args['l_idx'] + args['h_idx']
    shared_array[s_idx] = chroma


def solve_chroma_wrapper_fast(args):
    chroma = cl.solve_chroma_fast(**args)
    s_idx = args['h_sample_num'] * args['l_idx'] + args['h_idx']
    shared_array[s_idx] = chroma


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


def main_func():
    # make_gamut_boundary_lut_all()
    # make_gamut_boundary_lut_all_fast()
    make_focal_lut(
        outer_color_space_name=cs.BT2020,
        inner_color_space_name=cs.BT709)
    make_focal_lut(
        outer_color_space_name=cs.P3_D65,
        inner_color_space_name=cs.BT709)
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
