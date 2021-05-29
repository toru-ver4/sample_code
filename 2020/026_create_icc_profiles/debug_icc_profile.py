# -*- coding: utf-8 -*-
"""
debug code
==========

"""

# import standard libraries
import os
from multiprocessing import Pool, cpu_count

# import third-party libraries
import numpy as np
from colour.models import RGB_COLOURSPACES
from colour import RGB_to_RGB, Lab_to_XYZ, XYZ_to_RGB
import matplotlib.pyplot as plt

# import my libraries
import color_space as cs
import transfer_functions as tf
import test_pattern_generator2 as tpg
import icc_profile_xml_control as ipxc
from color_volume_boundary_data import calc_Lab_boundary_data
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def make_images(gamma_float=3.0):
    src_color_space = cs.ACES_AP0

    img = tpg.img_read_as_float(
        "./img/ColorChecker_All_ITU-R BT.709_D65_BT1886_Reverse.tiff")
    img_linear = img ** 2.4
    img_sRGB = tf.oetf(img_linear, tf.SRGB)
    ap1_img_linear = RGB_to_RGB(
        img_linear,
        RGB_COLOURSPACES[cs.BT709], RGB_COLOURSPACES[src_color_space])
    ap1_non_linear = ap1_img_linear ** (1/gamma_float)
    tpg.img_wirte_float_as_16bit_int("./img/ap0_img.png", ap1_non_linear)
    tpg.img_wirte_float_as_16bit_int("./img/sRGB.png", img_sRGB)


def make_ab_plane_color_image(
        ll=50.0, samples=1024, bg_color=0.5,
        xmin=-200, xmax=200, ymin=-200, ymax=200, cs_name=cs.BT709):
    """
    任意の Lightness の ab 平面の色画像をつくる
    """
    a_base = np.linspace(xmin, xmax, samples)
    b_base = np.linspace(ymin, ymax, samples)

    aa, bb = np.meshgrid(a_base, b_base)
    ll_array = np.ones_like(aa) * ll

    lab = np.dstack((ll_array, aa, bb))

    rgb = XYZ_to_RGB(
        Lab_to_XYZ(lab), cs.D65, cs.D65,
        RGB_COLOURSPACES[cs_name].XYZ_to_RGB_matrix)

    r_ok = (rgb[..., 0] >= 0) & (rgb[..., 0] <= 1.0)
    g_ok = (rgb[..., 1] >= 0) & (rgb[..., 1] <= 1.0)
    b_ok = (rgb[..., 2] >= 0) & (rgb[..., 2] <= 1.0)
    rgb_ok = (r_ok & g_ok) & b_ok

    rgb[~rgb_ok] = np.ones_like(rgb[~rgb_ok]) * bg_color
    rgb = rgb[::-1]

    rgb = rgb ** (1/2.4)

    return rgb


def plot_ab_plane_core(l_idx, lab, color_space_name):
    ll = lab[l_idx, 0, 0]
    aa = lab[l_idx, :, 1]
    bb = lab[l_idx, :, 2]
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title=f"CIELAB ab plane, {color_space_name}, L* = {ll:.3f}",
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
    # ax1.plot(aa, bb, '-k', label=f"{color_space_name} Gamut Boundary")
    rgb_img = make_ab_plane_color_image(
        ll=ll, samples=2048, bg_color=0.3,
        xmin=-200, xmax=200, ymin=-200, ymax=200, cs_name=color_space_name)
    ax1.imshow(
        rgb_img, extent=(-200, 200, -200, 200), aspect='auto')
    plt.legend(loc='upper left')
    fname = "/work/overuse/2020/026_icc_profile/img_seq/"
    fname += f"ab_plane_noline_{color_space_name}_{l_idx:04d}.png"
    print(fname)
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close(fig)


def plot_ab_plane_sequence(lab_bd, color_space_name):
    lab_data = lab_bd.get()

    l_num = lab_data.shape[0]
    block_num = 96

    for block_idx in range(block_num):
        block_data_num = l_num // block_num
        st_idx = block_idx * block_data_num
        if block_idx >= block_num - 1:
            ed_idx = l_num
        else:
            ed_idx = st_idx + block_data_num

        args = []
        for l_idx in range(st_idx, ed_idx):
            # l_idx = 128
            d = dict(
                l_idx=l_idx, lab=lab_data,
                color_space_name=color_space_name)
            args.append(d)
            # plot_ab_plane_core(**d)
            # break
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_plot_ab_plane_core, args)


def thread_wrapper_plot_ab_plane_core(args):
    plot_ab_plane_core(**args)


def make_cl_plane_color_image(
        hue=0.0, samples=1536, bg_color=0.5,
        xmin=-5, xmax=230, ymin=-5, ymax=105,
        cs_name=cs.BT2020):
    """
    任意の Hue の Chroma-Lightness 平面の色画像をつくる
    """
    c_base = np.linspace(xmin, xmax, samples)
    l_base = np.linspace(ymin, ymax, samples)

    cc, ll = np.meshgrid(c_base, l_base)
    aa = cc * np.cos(hue)
    bb = cc * np.sin(hue)

    lab = np.dstack((ll, aa, bb))

    rgb = XYZ_to_RGB(
        Lab_to_XYZ(lab), cs.D65, cs.D65,
        RGB_COLOURSPACES[cs_name].XYZ_to_RGB_matrix)

    r_ok = (rgb[..., 0] >= 0) & (rgb[..., 0] <= 1.0)
    g_ok = (rgb[..., 1] >= 0) & (rgb[..., 1] <= 1.0)
    b_ok = (rgb[..., 2] >= 0) & (rgb[..., 2] <= 1.0)
    rgb_ok = (r_ok & g_ok) & b_ok

    rgb[~rgb_ok] = np.ones_like(rgb[~rgb_ok]) * bg_color
    rgb = rgb[::-1]

    rgb = rgb ** (1/2.4)

    return rgb


def plot_cl_plane_core(h_idx, lab, color_space_name):
    ll = lab[:, h_idx, 0]
    aa = lab[:, h_idx, 1]
    bb = lab[:, h_idx, 2]
    cc = ((aa ** 2) + (bb ** 2)) ** 0.5
    cc_range_max = np.max(((lab[..., 1] ** 2) + (lab[..., 2] ** 2)) ** 0.5)\
        + 10
    hue = h_idx / (lab.shape[1] - 1) * np.pi * 2
    title = f"CIELAB cl plane, {color_space_name}, Hue={np.rad2deg(hue):.1f}°"
    xlim_min = 0
    xlim_max = cc_range_max
    ylim_min = -5
    ylim_max = 105
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title=title,
        graph_title_size=None,
        xlabel="Chroma", ylabel="Lightness",
        axis_label_size=None,
        legend_size=17,
        xlim=(xlim_min, xlim_max),
        ylim=(ylim_min, ylim_max),
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        return_figure=True)
    ax1.plot(cc, ll, '-k', label=f"{color_space_name} Gamut Boundary")
    rgb_img = make_cl_plane_color_image(
        hue=hue, samples=1536, bg_color=0.5,
        xmin=xlim_min, xmax=xlim_max, ymin=ylim_min, ymax=ylim_max,
        cs_name=color_space_name)
    ax1.imshow(
        rgb_img, extent=(xlim_min, xlim_max, ylim_min, ylim_max),
        aspect='auto')
    plt.legend(loc='lower right')
    fname = "/work/overuse/2020/026_icc_profile/img_seq/"
    fname += f"cl_plane_{color_space_name}_{h_idx:04d}.png"
    print(fname)
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close(fig)


def plot_cl_plane_sequence(lab_bd, color_space_name):
    lab_data = lab_bd.get()

    h_num = lab_data.shape[1]
    block_num = 16

    for block_idx in range(block_num):
        block_data_num = h_num // block_num
        st_idx = block_idx * block_data_num
        if block_idx >= block_num - 1:
            ed_idx = h_num
        else:
            ed_idx = st_idx + block_data_num

        args = []
        for h_idx in range(st_idx, ed_idx):
            d = dict(
                h_idx=h_idx, lab=lab_data,
                color_space_name=color_space_name)
            args.append(d)
            # plot_cl_plane_core(**d)
            # break
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_plot_cl_plane_core, args)


def thread_wrapper_plot_cl_plane_core(args):
    plot_cl_plane_core(**args)


def plot_lab_3d_sequence(lab_bd):
    pass


def debug_lab_color_space_plot(color_space_name=cs.BT2020):
    l_num = 4096
    h_num = 1024

    lab_bd = calc_Lab_boundary_data(
        color_space_name=cs.BT2020, white=cs.D65,
        l_num=l_num, h_num=h_num, overwirte_lut=False)

    plot_ab_plane_sequence(lab_bd=lab_bd, color_space_name=color_space_name)
    # plot_cl_plane_sequence(lab_bd=lab_bd, color_space_name=color_space_name)
    plot_lab_3d_sequence(lab_bd)


def main_func():
    # convert color checker from BT709 to ACES AP0
    make_images(gamma_float=3.5)

    # create "Gamma3.5_ACES-AP0_D65"
    ipxc.create_simple_power_gamma_profile(
        gamma=3.5, src_white=cs.D65,
        src_primaries=cs.get_primaries(cs.ACES_AP0),
        desc_str="Gamma3.5_ACES-AP0_D65",
        cprt_str="Copyright 2020 Toru Yoshihara.",
        output_name="Gamma3.5_ACES-AP0_D65.xml")

    # create "Gamma2.4_BT.709_D65"
    ipxc.create_simple_power_gamma_profile(
        gamma=2.4, src_white=cs.D65,
        src_primaries=cs.get_primaries(cs.BT709),
        desc_str="Gamma2.4_BT.709_D65",
        cprt_str="Copyright 2020 Toru Yoshihara.",
        output_name="Gamma2.4_BT.709_D65.xml")

    # create "Gamma2.4_BT.2020_D65"
    ipxc.create_simple_power_gamma_profile(
        gamma=2.4, src_white=cs.D65,
        src_primaries=cs.get_primaries(cs.BT2020),
        desc_str="Gamma2.4_BT.2020_D65",
        cprt_str="Copyright 2020 Toru Yoshihara.",
        output_name="Gamma2.4_BT.2020_D65.xml")

    # create "Gamma2.4_BT.2020_D65"
    ipxc.create_simple_power_gamma_profile(
        gamma=2.4, src_white=cs.D65,
        src_primaries=cs.get_primaries(cs.P3_D65),
        desc_str="Gamma2.4_P3_D65",
        cprt_str="Copyright 2020 Toru Yoshihara.",
        output_name="Gamma2.4_DCI-P3_D65.xml")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
    # debug_lab_color_space_plot(cs.BT2020)
