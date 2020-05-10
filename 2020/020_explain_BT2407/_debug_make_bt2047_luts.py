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
from colour import RGB_COLOURSPACES
from colour.models import BT709_COLOURSPACE
import cv2

# import my libraries
import plot_utility as pu
import color_space as cs
from bt2407_parameters import L_SAMPLE_NUM_MAX, H_SAMPLE_NUM_MAX,\
    GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE, GAMUT_BOUNDARY_LUT_HUE_SAMPLE,\
    get_gamut_boundary_lut_name, get_l_cusp_name, get_focal_name,\
    get_chroma_map_lut_name
from bt2047_gamut_mapping import get_chroma_lightness_val_specfic_hue,\
    calc_chroma_lightness_using_length_from_l_focal,\
    calc_chroma_lightness_using_length_from_c_focal
from make_bt2047_luts import calc_value_from_hue_1dlut,\
    calc_chroma_map_degree2, calc_l_cusp_specific_hue, calc_cusp_in_lc_plane
import test_pattern_generator2 as tpg


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

    fig, ax1 = pu.plot_1_graph(
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
        return_figure=True)
    # ax1.plot(a, b, label="L*={:.03f}".format(idx * 100 / (l_sample_num - 1)))
    ax1.patch.set_facecolor("#B0B0B0")
    # ax1.scatter(a, b, c=rgb)
    ax1.plot(a, b, '-k')
    ax1.scatter(inner_lab[..., 1], inner_lab[..., 2], c=inner_rgb, s=7.5)
    # plt.legend(loc='upper left')
    plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    print("plot l_idx={}".format(idx))
    # plt.show()
    plt.close(fig)


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


def plot_bt709_p3_bt2020_gamut_boundary():
    visualization_ab_plane_fill_color(
        test_sample_grid_num=144, color_space_name=cs.BT709,
        l_sample_num=GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE,
        h_sample_num=GAMUT_BOUNDARY_LUT_HUE_SAMPLE)
    visualization_ab_plane_fill_color(
        test_sample_grid_num=144, color_space_name=cs.BT2020,
        l_sample_num=GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE,
        h_sample_num=GAMUT_BOUNDARY_LUT_HUE_SAMPLE)
    visualization_ab_plane_fill_color(
        test_sample_grid_num=144, color_space_name=cs.P3_D65,
        l_sample_num=GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE,
        h_sample_num=GAMUT_BOUNDARY_LUT_HUE_SAMPLE)


def plot_simple_cl_plane(
        hue=np.deg2rad(30),
        inner_color_space_name=cs.BT709,
        outer_color_space_name=cs.BT2020,
        chroma_min=-5, chroma_max=220, ll_min=0, ll_max=100):
    """
    ブログでの説明用にシンプルな Chroma Lightness平面をプロット
    """
    cl_inner = get_chroma_lightness_val_specfic_hue(
        hue=hue,
        lh_lut_name=get_gamut_boundary_lut_name(inner_color_space_name))
    cl_outer =\
        get_chroma_lightness_val_specfic_hue(
            hue=hue,
            lh_lut_name=get_gamut_boundary_lut_name(outer_color_space_name))

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title=f"HUE = {hue/2/np.pi*360:.1f}°",
        graph_title_size=None,
        xlabel="Chroma",
        ylabel="Lightness",
        axis_label_size=None,
        legend_size=17,
        xlim=[chroma_min, chroma_max],
        ylim=[ll_min, ll_max],
        xtick=[20 * x for x in range(12)],
        ytick=[x * 10 for x in range(11)],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        return_figure=True)
    ax1.patch.set_facecolor("#E0E0E0")
    in_color = "#707070"
    ou_color = "#000000"

    # gamut boundary
    ax1.plot(
        cl_inner[..., 0], cl_inner[..., 1], '--', c=in_color,
        label=inner_color_space_name)
    ax1.plot(
        cl_outer[..., 0], cl_outer[..., 1], c=ou_color,
        label=outer_color_space_name)

    graph_name = f"./figures/simple_cl_plane_HUE_{hue/2/np.pi*360:.1f}.png"
    plt.legend(loc='upper right')
    plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close(fig)


def plot_filled_cl_plane(
        hue=np.deg2rad(30), hue_idx=0,
        color_space_name=cs.BT709,
        chroma_min=-5, chroma_max=140, ll_min=0, ll_max=100):
    """
    CL平面を塗りつぶす！
    """
    rgb_img = make_ch_plane_color_image(
        hue=hue, samples=1024, bg_color=0.5,
        xmin=chroma_min, xmax=chroma_max, ymin=ll_min, ymax=ll_max,
        cs_name=color_space_name)
    cl_inner = get_chroma_lightness_val_specfic_hue(
        hue=hue,
        lh_lut_name=get_gamut_boundary_lut_name(color_space_name))

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 8),
        graph_title=f"HUE = {np.rad2deg(hue):.1f}°",
        graph_title_size=None,
        xlabel="Chroma",
        ylabel="Lightness",
        axis_label_size=None,
        legend_size=17,
        xlim=[chroma_min, chroma_max],
        ylim=[ll_min, ll_max],
        xtick=[20 * x for x in range(int(chroma_max // 20 + 1))],
        ytick=[x * 10 for x in range(int(ll_max // 10) + 1)],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        return_figure=True)
    in_color = "#000000"

    # gamut boundary
    ax1.plot(
        cl_inner[..., 0], cl_inner[..., 1], '-', c=in_color, lw=3,
        label=color_space_name)
    ax1.imshow(
        rgb_img, extent=(chroma_min, chroma_max, ll_min, ll_max),
        aspect='auto')

    graph_name = f"./cl_plane_seq/filled_cl_plane_"\
        + f"{color_space_name}_{hue_idx:04d}.png"
    plt.legend(loc='upper right')
    plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close(fig)


def make_ch_plane_color_image(
        hue=np.deg2rad(40), samples=1024, bg_color=0.5,
        xmin=0, xmax=220, ymin=0, ymax=100, cs_name=cs.BT709):
    """
    任意の Hue の Chroma-Lightness 平面の色画像をつくる
    """
    chroma = np.linspace(xmin, xmax, samples)
    lightness = np.linspace(ymin, ymax, samples)
    cc, ll = np.meshgrid(chroma, lightness)
    aa = cc * np.cos(hue)
    bb = cc * np.sin(hue)

    lab = np.dstack((ll, aa, bb))

    rgb = XYZ_to_RGB(
        Lab_to_XYZ(lab), tpg.D65_WHITE, tpg.D65_WHITE,
        RGB_COLOURSPACES[cs_name].XYZ_to_RGB_matrix)

    r_ok = (rgb[..., 0] >= 0) & (rgb[..., 0] <= 1.0)
    g_ok = (rgb[..., 1] >= 0) & (rgb[..., 1] <= 1.0)
    b_ok = (rgb[..., 2] >= 0) & (rgb[..., 2] <= 1.0)
    rgb_ok = (r_ok & g_ok) & b_ok

    rgb[~rgb_ok] = np.ones_like(rgb[~rgb_ok]) * bg_color
    rgb = rgb[::-1]

    rgb = rgb ** (1/2.4)

    return rgb


def thread_wapper_cl_plen_filled_color(args):
    plot_filled_cl_plane(**args)


def make_cl_plane_filled_color(hue_sample=5):
    hue_list = np.deg2rad(np.linspace(0, 360, hue_sample, endpoint=False))
    lut_name = get_gamut_boundary_lut_name(cs.BT2020)
    lh_lut = np.load(lut_name)
    chroma_max = np.ceil(np.max(lh_lut) / 10) * 10

    color_space_name_list = [cs.BT709, cs.P3_D65, cs.BT2020]
    for color_space_name in color_space_name_list:
        args = []
        for idx, hue in enumerate(hue_list):
            # plot_filled_cl_plane(
            #     hue=hue, hue_idx=idx,
            #     color_space_name=color_space_name,
            #     chroma_min=0, chroma_max=chroma_max, ll_min=0, ll_max=100)
            d = dict(
                hue=hue, hue_idx=idx,
                color_space_name=color_space_name,
                chroma_min=0, chroma_max=chroma_max, ll_min=0, ll_max=100
            )
            args.append(d)
        with Pool(cpu_count()) as pool:
            pool.map(thread_wapper_cl_plen_filled_color, args)


def make_example_patch():
    size = 200

    rgb_list = np.array([
        [245, 85, 57], [219, 109, 83], [214, 50, 30], [216, 78, 53]])
    fname_list = [
        "./figures/org_r.png", "./figures/fixed_lumi.png",
        "./figures/fixed_ch.png", "./figures/balance.png"]

    img_list = [np.ones((size, size, 3), dtype=np.uint8) * rgb
                for rgb in rgb_list]

    for name, img in zip(fname_list, img_list):
        cv2.imwrite(name, img[..., ::-1])


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

    graph_name = f"./cl_plane_seq/cmap_lut_check_{focal_type}_{idx:04d}.png"
    plt.legend(loc='upper right')
    plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    # plt.show()


def _check_chroma_map_lut_data(
        h_idx,
        outer_color_space_name=cs.BT2020,
        inner_color_space_name=cs.BT709):
    hue = h_idx / (GAMUT_BOUNDARY_LUT_HUE_SAMPLE - 1) * 2 * np.pi

    # とりあえず L*C* 平面のポリゴン準備
    cl_inner = get_chroma_lightness_val_specfic_hue(
        hue, get_gamut_boundary_lut_name(inner_color_space_name))
    cl_outer = get_chroma_lightness_val_specfic_hue(
        hue, get_gamut_boundary_lut_name(outer_color_space_name))

    # cusp 準備
    lh_inner_lut = np.load(
        get_gamut_boundary_lut_name(inner_color_space_name))
    inner_cusp = calc_cusp_in_lc_plane(hue, lh_inner_lut)
    lh_outer_lut = np.load(
        get_gamut_boundary_lut_name(outer_color_space_name))
    inner_cusp = calc_cusp_in_lc_plane(hue, lh_inner_lut)
    outer_cusp = calc_cusp_in_lc_plane(hue, lh_outer_lut)

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
    l_cusp = calc_value_from_hue_1dlut(hue, l_cusp_lut)
    l_focal = calc_value_from_hue_1dlut(hue, l_focal_lut)
    c_focal = calc_value_from_hue_1dlut(hue, c_focal_lut)

    # Chroma Mapping の距離のデータ
    st_degree_l, ed_degree_l, st_degree_c, ed_degree_c =\
        calc_chroma_map_degree2(l_focal, c_focal, inner_cusp)
    cmap_lut_c = np.load(
        get_chroma_map_lut_name(
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name,
            focal_type="Cfocal"))
    cmap_lut_l = np.load(
        get_chroma_map_lut_name(
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name,
            focal_type="Lfocal"))

    degree_sample = cmap_lut_l.shape[1]
    degree_l = np.linspace(st_degree_l, ed_degree_l, degree_sample)
    degree_c = np.linspace(st_degree_c, ed_degree_c, degree_sample)

    icn_x_l, icn_y_l = calc_chroma_lightness_using_length_from_l_focal(
        distance=cmap_lut_l[h_idx], degree=degree_l, l_focal=l_focal)
    icn_x_c, icn_y_c = calc_chroma_lightness_using_length_from_c_focal(
        distance=cmap_lut_c[h_idx], degree=degree_c, c_focal=c_focal)

    _debug_plot_check_chroma_map_lut_specific_hue(
        hue, cl_inner, cl_outer, l_cusp, inner_cusp, outer_cusp,
        l_cusp, l_focal, c_focal,
        icn_x=icn_x_l, icn_y=icn_y_l,
        focal_type="L_focal", idx=h_idx)
    _debug_plot_check_chroma_map_lut_specific_hue(
        hue, cl_inner, cl_outer, l_cusp, inner_cusp, outer_cusp,
        l_cusp, l_focal, c_focal,
        icn_x=icn_x_c, icn_y=icn_y_c,
        focal_type="C_focal", idx=h_idx)


def main_func():
    # 確認
    # plot_bt709_p3_bt2020_gamut_boundary()
    # plot_simple_cl_plane(hue=np.deg2rad(270))
    # make_cl_plane_filled_color(hue_sample=2048)
    # make_example_patch()
    # _check_chroma_map_lut_data(0, cs.BT2020, cs.BT709)
    # _check_chroma_map_lut_data(256, cs.BT2020, cs.BT709)
    # _check_chroma_map_lut_data(512, cs.BT2020, cs.BT709)
    # _check_chroma_map_lut_data(768, cs.BT2020, cs.BT709)
    _check_chroma_map_lut_data(0, cs.P3_D65, cs.BT709)
    _check_chroma_map_lut_data(256, cs.P3_D65, cs.BT709)
    _check_chroma_map_lut_data(512, cs.P3_D65, cs.BT709)
    _check_chroma_map_lut_data(768, cs.P3_D65, cs.BT709)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
