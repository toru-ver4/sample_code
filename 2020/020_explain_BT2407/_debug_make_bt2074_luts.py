# -*- coding: utf-8 -*-
"""
デバッグ用のコード集
====================

"""

# import standard libraries
import os

# import third-party libraries
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, cpu_count
from colour import LUT3D, RGB_to_XYZ, XYZ_to_Lab, Lab_to_XYZ, XYZ_to_RGB

# import my libraries
import plot_utility as pu
import color_space as cs
from bt2407_parameters import L_SAMPLE_NUM_MAX, H_SAMPLE_NUM_MAX,\
    GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE, GAMUT_BOUNDARY_LUT_HUE_SAMPLE,\
    get_gamut_boundary_lut_name


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def plot_and_save_ab_plane(idx, data, l_sample_num, h_sample_num):
    rad = np.linspace(0, 2 * np.pi, h_sample_num)
    a = data * np.cos(rad)
    b = data * np.sin(rad)
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="CIELAB Plane",
        graph_title_size=None,
        xlabel="a*", ylabel="b*",
        axis_label_size=None,
        legend_size=17,
        xlim=(-200, 200),
        ylim=(-200, 200),
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(a, b, label="L*={:.03f}".format(idx * 100 / (l_sample_num - 1)))
    plt.legend(loc='upper left')
    print("plot l_idx={}".format(idx))
    plt.show()


def plot_and_save_ab_plane_fill_color(
        idx, data, inner_rgb, inner_lab, l_sample_num, h_sample_num,
        color_space_name=cs.BT709):
    graph_name = f"./ab_plane_seq/debug_boundary_lut_fill_color_"\
        + f"L_{l_sample_num}_{color_space_name}_{idx:04d}.png"
    rad = np.linspace(0, 2 * np.pi, h_sample_num)
    a = data * np.cos(rad)
    b = data * np.sin(rad)
    large_l = np.ones_like(a) * (idx * 100) / (l_sample_num - 1)
    lab = np.dstack((large_l, a, b)).reshape((h_sample_num, 3))
    large_xyz = Lab_to_XYZ(lab)
    rgb = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65, cs.get_xyz_to_rgb_matrix(color_space_name))
    rgb = np.clip(rgb, 0.0, 1.0) ** (1/2.4)

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="CIELAB Plane L*={:.03f}".format(
            idx * 100 / (l_sample_num - 1)) + f", {color_space_name}",
        graph_title_size=None,
        xlabel="a*", ylabel="b*",
        axis_label_size=None,
        legend_size=17,
        xlim=(-200, 200),
        ylim=(-200, 200),
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    # ax1.plot(a, b, label="L*={:.03f}".format(idx * 100 / (l_sample_num - 1)))
    ax1.patch.set_facecolor("#B0B0B0")
    # ax1.scatter(a, b, c=rgb)
    ax1.plot(a, b, '-k')
    ax1.scatter(inner_lab[..., 1], inner_lab[..., 2], c=inner_rgb, s=7.5)
    # plt.legend(loc='upper left')
    plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    print("plot l_idx={}".format(idx))
    # plt.show()


def visualization_ab_plane_fill_color(
        test_sample_grid_num=64, color_space_name=cs.BT709,
        l_sample_num=L_SAMPLE_NUM_MAX, h_sample_num=H_SAMPLE_NUM_MAX):
    """
    ab plane を L* = 0～100 で静止画にして吐く。
    後で Resolve で動画にして楽しもう！
    """
    npy_name = get_gamut_boundary_lut_name(
        color_space_name=color_space_name,
        luminance_sample_num=l_sample_num, hue_sample_num=h_sample_num)
    calc_data = np.load(npy_name)
    delta_l = 0.001 * 100
    gamma = 2.4
    rgb = LUT3D.linear_table(
        test_sample_grid_num).reshape((1, test_sample_grid_num ** 3, 3))\
        ** (gamma)
    xyz = RGB_to_XYZ(
        rgb, cs.D65, cs.D65, cs.get_rgb_to_xyz_matrix(color_space_name))
    lab = XYZ_to_Lab(xyz)

    args = []
    l_list = np.linspace(0, 100, l_sample_num)

    for l_idx, l_val in enumerate(l_list):
        ok_idx = (l_val - delta_l <= lab[:, :, 0])\
            & (lab[:, :, 0] < l_val + delta_l)
        d = dict(
            idx=l_idx, data=calc_data[l_idx], inner_rgb=rgb[ok_idx],
            inner_lab=lab[ok_idx], l_sample_num=l_sample_num,
            h_sample_num=h_sample_num, color_space_name=color_space_name)
        args.append(d)
        # plot_and_save_ab_plane_fill_color(**d)
    with Pool(cpu_count()) as pool:
        pool.map(thread_wrapper_visualization_ab_plane_fill_color, args)


def thread_wrapper_visualization_ab_plane_fill_color(args):
    return plot_and_save_ab_plane_fill_color(**args)


def main_func():
    # 確認
    # visualization_ab_plane_fill_color(
    #     test_sample_grid_num=192, color_space_name=cs.BT709,
    #     l_sample_num=GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE,
    #     h_sample_num=GAMUT_BOUNDARY_LUT_HUE_SAMPLE)
    # visualization_ab_plane_fill_color(
    #     test_sample_grid_num=192, color_space_name=cs.BT2020,
    #     l_sample_num=GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE,
    #     h_sample_num=GAMUT_BOUNDARY_LUT_HUE_SAMPLE)
    # visualization_ab_plane_fill_color(
    #     test_sample_grid_num=192, color_space_name=cs.P3_D65,
    #     l_sample_num=GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE,
    #     h_sample_num=GAMUT_BOUNDARY_LUT_HUE_SAMPLE)

    # 任意の L* の a*b*平面用の boundary データ計算
    # lstar = 50
    # h_sample_num = 256
    # fname = f"Chroma_L_{lstar}_BT709_h_{h_sample_num}.npy"
    # chroma = np.load(fname)
    # plot_and_save_ab_plane(1, chroma, 0, h_sample_num)
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
