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
from scipy import signal, interpolate

# import my libraries
import plot_utility as pu
import interpolate_cielab_data as icd

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


BT709_BOUNDARY = "./boundary_data/Chroma_BT709_l_256_h_256.npy"
BT2020_BOUNDARY = "./boundary_data/Chroma_BT2020_l_256_h_256.npy"
L_CUSP_NAME = "./L_CUSP.npy"
L_FOCAL_NAME = "./L_FOCAL.npy"
C_FOCAL_NAME = "./C_FOCAL.npy"

L_SEARCH_SAMPLE = 256
C_SEARCH_SAMPLE = 256

DIPS_150_SAMPLE_ST = int(0.32 * C_SEARCH_SAMPLE)
DIPS_150_SAMPLE_ED = int(0.6 * C_SEARCH_SAMPLE)
DIPS_300_SAMPLE_ST = int(0.75 * C_SEARCH_SAMPLE)
DIPS_300_SAMPLE_ED = int(0.9 * C_SEARCH_SAMPLE)

# BT.2407 の FIGURE A2-4 を見ると 240° くらいで終わってるので…
L_FOCAL_240_INDEX = int(240 / 360 * C_SEARCH_SAMPLE)


def plot_lc_plane_specific_hue(hue=0/360*2*np.pi):
    """
    とりあえず L*C* 平面をプロット
    """
    sample_num = 1024
    lstar = np.linspace(0, 100, sample_num)
    hue_list = np.ones_like(lstar) * hue
    lh = np.dstack([lstar, hue_list])
    lut_bt709 = np.load(BT709_BOUNDARY)
    lut_bt2020 = np.load(BT2020_BOUNDARY)
    chroma_bt709 = icd.bilinear_interpolation(lh, lut_bt709)
    chroma_bt2020 = icd.bilinear_interpolation(lh, lut_bt2020)

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title=f"HUE = {hue/2/np.pi*360:.1f}°",
        graph_title_size=None,
        xlabel="Chroma",
        ylabel="Lightness",
        axis_label_size=None,
        legend_size=17,
        xlim=[-6, 200],
        ylim=[-3, 103],
        xtick=[x * 20 for x in range(11)],
        ytick=[x * 10 for x in range(11)],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(chroma_bt709, lstar, label="BT.709")
    ax1.plot(chroma_bt2020, lstar, label="BT.2020")
    graph_name = f"./figure/HUE = {hue/2/np.pi*360:.1f}.png"
    plt.legend(loc='lower right')
    plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    # plt.show()


def _debug_plot_lc_plane(x, y, **kwargs):
    """
    デバッグ用に L*C*平面をプロット。
    """
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title=f"Debug Plot",
        graph_title_size=None,
        xlabel="Chroma",
        ylabel="Lightness",
        axis_label_size=None,
        legend_size=17,
        xlim=[-6, 200],
        ylim=[-3, 103],
        xtick=[x * 20 for x in range(11)],
        ytick=[x * 10 for x in range(11)],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(x, y, label="-")
    # graph_name = f"./figure/HUE = {hue/2/np.pi*360:.1f}.png"
    # plt.legend(loc='lower right')
    # plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def _debug_plot_lc_plane_with_cups(
        inner_lh, outer_lh, inner_cusp, outer_cusp, lcusp):
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title=f"Debug Plot",
        graph_title_size=None,
        xlabel="Chroma",
        ylabel="Lightness",
        axis_label_size=None,
        legend_size=17,
        xlim=[-6, 200],
        ylim=[-3, 103],
        xtick=[x * 20 for x in range(11)],
        ytick=[x * 10 for x in range(11)],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(inner_lh[..., 1], inner_lh[..., 0], c=pu.RED, label="inner gamut")
    ax1.plot(
        outer_lh[..., 1], outer_lh[..., 0], c=pu.BLUE, label="outer gamut")
    ax1.plot(
        inner_cusp[1], inner_cusp[0], 'o', ms=10, c=pu.RED, label="inner cusp")
    ax1.plot(outer_cusp[1], outer_cusp[0], 'o',
             ms=10, c=pu.BLUE, label="outer cusp")
    ax1.plot(lcusp[1], lcusp[0], 's', ms=10, c='k', label="L_cusp")
    ax1.plot([lcusp[1], outer_cusp[1]], [lcusp[0], outer_cusp[0]], 'k--', lw=1)
    # graph_name = f"./figure/HUE = {hue/2/np.pi*360:.1f}.png"
    plt.legend(loc='lower right')
    # plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def check_lc_plane():
    for hue in np.linspace(0, 360, 12, endpoint=False):
        plot_lc_plane_specific_hue(hue/360*2*np.pi)


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


def _debug_calc_interpolated_lstar_chroma_value(hue, lh_lut):
    l_sample = np.linspace(0, 100, L_SEARCH_SAMPLE)
    h_sample = np.ones_like(l_sample) * hue
    lh_sample = np.dstack((l_sample, h_sample))
    chroma_for_each_l = icd.bilinear_interpolation(
        lh=lh_sample, lut2d=lh_lut)

    return np.dstack((l_sample, chroma_for_each_l))[0]


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
    l_sample = np.linspace(0, 100, L_SEARCH_SAMPLE)
    h_sample = np.ones_like(l_sample) * hue
    lh_sample = np.dstack((l_sample, h_sample))
    chroma_for_each_l = icd.bilinear_interpolation(lh=lh_sample, lut2d=lh_lut)
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

    # debug
    # inner_lh = _debug_calc_interpolated_lstar_chroma_value(hue, inner_lut)
    # outer_lh = _debug_calc_interpolated_lstar_chroma_value(hue, outer_lut)
    # _debug_plot_lc_plane_with_cups(
    #     inner_lh, outer_lh, inner_cusp, outer_cusp, lcusp)

    return lcusp[0]


def _debug_plot_l_cusp(l_cusp, l_focal, dips_150, dips_300, low_pass):

    x = np.linspace(0, 360, len(l_cusp))
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title=f"Debug Plot",
        graph_title_size=None,
        xlabel="Hue",
        ylabel="Lightness",
        axis_label_size=None,
        legend_size=17,
        xlim=[-10, 370],
        ylim=None,
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
    plt.savefig("./figure/L_focal.png", bbox_inches='tight', pad_inches=0.1)
    plt.show()


def low_pass_filter(x, nn=4, wn=0.25):
    b1, a1 = signal.butter(nn, wn, "low")
    result = signal.filtfilt(b1, a1, x)

    return result


def low_pass_filter2(x, nn=4, wn=0.25):
    b1, a1 = signal.bessel(nn, wn, "low")
    result = signal.filtfilt(b1, a1, x)

    return result


def get_dips_value_around_150(l_cusp):
    """
    135°付近の凹みの L* 値および、それを指す Hue の Index を計算する。
    """
    dips_150 = np.min(l_cusp[DIPS_150_SAMPLE_ST:DIPS_150_SAMPLE_ED])
    dips_150_idx = np.argmin(l_cusp[DIPS_150_SAMPLE_ST:DIPS_150_SAMPLE_ED])
    return dips_150, dips_150_idx


def get_dips_value_around_300(l_cusp):
    """
    300°付近の凹みの L* 値および、それを指す Hue の Index を計算する。
    """
    dips_300 = np.min(l_cusp[DIPS_300_SAMPLE_ST:DIPS_300_SAMPLE_ED])
    dips_300_idx = np.argmin(l_cusp[DIPS_300_SAMPLE_ST:DIPS_300_SAMPLE_ED])
    dips_300_idx += DIPS_300_SAMPLE_ST
    return dips_300, dips_300_idx


def calc_l_cusp():
    inner_lut = np.load(BT709_BOUNDARY)
    outer_lut = np.load(BT2020_BOUNDARY)
    l_cusp = []
    h_sample = C_SEARCH_SAMPLE
    hue_list = np.linspace(0, 2*np.pi, h_sample)

    for hue in hue_list:
        lll = calc_l_cusp_specific_hue(hue, inner_lut, outer_lut)
        l_cusp.append(lll)
    l_cusp = np.array(l_cusp)

    return l_cusp


def calc_l_focal(l_cusp):
    """
    l_cusp に修正を加えた l_focal を求める

    1. min, max の設定。np.clip() で dips300, dips150 の範囲内に制限
    2. 240°～300° にかけて緩やかにスロープさせる
    """
    dips_150, _ = get_dips_value_around_150(l_cusp)
    dips_300, dips_300_idx = get_dips_value_around_300(l_cusp)
    decrement_sample = dips_300_idx - L_FOCAL_240_INDEX + 1
    decrement_data = np.linspace(dips_150, dips_300, decrement_sample)
    l_cusp_low_pass = low_pass_filter2(l_cusp, nn=4, wn=0.3)
    l_focal = np.clip(l_cusp_low_pass, dips_300, dips_150)
    l_focal[L_FOCAL_240_INDEX:dips_300_idx + 1] = decrement_data
    l_focal[dips_300_idx:] = dips_300

    _debug_plot_l_cusp(l_cusp, l_focal, dips_150, dips_300, l_cusp_low_pass)

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


def _debug_plot_lc_plane_with_c_focal(
        inner_lh, outer_lh, inner_cusp, outer_cusp, c_focal, hue):
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title=f"Debug Plot HUE={hue/2/np.pi*360:.1f}",
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
    ax1.plot(inner_lh[..., 1], inner_lh[..., 0], c=pu.RED, label="inner gamut")
    ax1.plot(
        outer_lh[..., 1], outer_lh[..., 0], c=pu.BLUE, label="outer gamut")
    ax1.plot(
        inner_cusp[1], inner_cusp[0], 'o', ms=10, c=pu.RED, label="inner cusp")
    ax1.plot(outer_cusp[1], outer_cusp[0], 'o',
             ms=10, c=pu.BLUE, label="outer cusp")
    ax1.plot(c_focal[1], c_focal[0], 's', ms=10, c='k', label="C_focal")
    if c_focal[1] < 0:
        ax1.plot([c_focal[1], outer_cusp[1]], [c_focal[0], outer_cusp[0]],
                 'k--', lw=1)
        plt.legend(loc='upper left')

    else:
        ax1.plot([c_focal[1], inner_cusp[1]], [c_focal[0], inner_cusp[0]],
                 'k--', lw=1)
        plt.legend(loc='upper right')
    graph_name = f"./figure/C_focal_HUE_{hue/2/np.pi*360:.1f}.png"
    plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    # plt.show()


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

    # debug
    # if c_focal[1] == 0:
    #     inner_lh = _debug_calc_interpolated_lstar_chroma_value(hue, inner_lut)
    #     outer_lh = _debug_calc_interpolated_lstar_chroma_value(hue, outer_lut)
    #     _debug_plot_lc_plane_with_c_focal(
    #         inner_lh, outer_lh, inner_cusp, outer_cusp, c_focal, hue)

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


def _debug_plot_c_focal(c_focal):
    x = np.linspace(0, 360, len(c_focal))
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title=f"Debug Plot",
        graph_title_size=None,
        xlabel="Hue",
        ylabel="Chroma",
        axis_label_size=None,
        legend_size=17,
        xlim=[-10, 370],
        ylim=None,
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

    # ろーぱすふぃるたー
    low_pass = low_pass_filter2(c_focal_interp, nn=4, wn=0.3)
    ax1.plot(x, low_pass, '-', lw=3, label="Apply LPF")
    plt.legend(loc='upper right')
    plt.savefig("./figure/C_focal.png", bbox_inches='tight', pad_inches=0.1)
    plt.show()


def calc_c_focal():
    inner_lut = np.load(BT709_BOUNDARY)
    outer_lut = np.load(BT2020_BOUNDARY)

    c_focal = []
    h_sample = C_SEARCH_SAMPLE
    hue_list = np.linspace(0, 2*np.pi, h_sample)

    for idx, hue in enumerate(hue_list):
        lll = calc_c_focal_specific_hue(hue, inner_lut, outer_lut)
        c_focal.append(lll)
        # break
    c_focal = np.abs(np.array(c_focal))
    _debug_plot_c_focal(c_focal)

    # 色々と問題があるので補間とかLPFとか処理する
    c_focal_interp = interpolate_where_value_is_zero(hue_list, c_focal)
    c_focal_lpf = low_pass_filter2(c_focal_interp, nn=4, wn=0.3)

    return c_focal_lpf


def main_func():
    l_cusp = calc_l_cusp()
    np.save(L_CUSP_NAME, l_cusp)
    l_focal = calc_l_focal(l_cusp)
    np.save(L_FOCAL_NAME, l_focal)
    c_focal = calc_c_focal()
    np.save(C_FOCAL_NAME, c_focal)

    # l_cusp = np.load(L_CUSP_NAME)
    # l_focal = np.load(L_FOCAL_NAME)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # check_lc_plane()
    main_func()
