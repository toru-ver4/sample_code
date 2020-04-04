# -*- coding: utf-8 -*-
"""
Lightness Mapping の実行
=====================================

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from colour import Lab_to_XYZ, XYZ_to_RGB
from colour.models import BT2020_COLOURSPACE

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
    src_sample = 11
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
        hue, cl_inner, cl_outer, lcusp, inner_cusp, outer_cusp, src_cl_value,
        l_cusp, l_focal, c_focal):
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
    ax1.scatter(src_cl_value[..., 0], src_cl_value[..., 1], s=150,
                c=src_cl_rgb[0], label="src", zorder=3)
    graph_name = f"./figure/lm_test_HUE_{hue/2/np.pi*360:.1f}.png"
    plt.legend(loc='upper right')
    plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def calc_value_from_hue_1dlut(val, lut):
    x = np.linspace(0, 2*np.pi, len(lut))
    f = interpolate.interp1d(x, lut)
    y = f(val)

    return y


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

    _debug_plot_lightness_mapping_specific_hue(
        hue, cl_inner, cl_outer, lcusp, inner_cusp, outer_cusp, src_cl_value,
        l_cusp, l_focal, c_focal)


def main_func():
    # とりあえず、任意の Hue において一発 Lightness Mapping してみる
    _try_lightness_mapping_specific_hue(hue=00/360*2*np.pi)
    _try_lightness_mapping_specific_hue(hue=90/360*2*np.pi)
    _try_lightness_mapping_specific_hue(hue=180/360*2*np.pi)
    _try_lightness_mapping_specific_hue(hue=270/360*2*np.pi)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
