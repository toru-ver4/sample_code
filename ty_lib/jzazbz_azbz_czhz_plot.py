# -*- coding: utf-8 -*-
"""

```
"""

# import standard libraries
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

# import third-party libraries
import numpy as np
from colour import Lab_to_XYZ, XYZ_to_RGB
from colour.models import RGB_COLOURSPACES
from colour.utilities import tstack

# import my libraries
import plot_utility as pu
import color_space as cs
import transfer_functions as tf
from create_gamut_booundary_lut import JZAZBZ_CHROMA_MAX, is_out_of_gamut_rgb,\
    create_jzazbz_gamut_boundary_lut_type2,\
    make_jzazbz_gb_lut_fname_methodb_b, make_jzazbz_gb_lut_fname_method_c,\
    TyLchLut
from color_space_plane_plot import create_valid_jzazbz_ab_plane_image_st2084,\
    create_valid_jzazbz_cj_plane_image_st2084

from jzazbz import jzazbz_to_large_xyz, jzczhz_to_jzazbz

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def debug_plot_azbz_plane_with_interpolation(
        hue_sample=256, lightness_sample=256, j_num_intp=256,
        color_space_name=cs.BT2020, luminance=10000):

    lut_name = make_jzazbz_gb_lut_fname_method_c(
        color_space_name=color_space_name, luminance=luminance,
        lightness_num=lightness_sample, hue_num=hue_sample)
    bg_lut = TyLchLut(np.load(lut_name))
    j_max = bg_lut.ll_max

    j_num = j_num_intp

    total_process_num = j_num
    block_process_num = int(cpu_count() / 2 + 0.999)
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            j_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={j_idx}")  # User
            if j_idx >= total_process_num:                         # User
                break
            # j_idx = j_num - 1
            d = dict(
                bg_lut_name=lut_name, j_idx=j_idx,
                j_val=j_idx/(j_num-1) * j_max,
                color_space_name=color_space_name,
                maximum_luminance=luminance)
            # plot_ab_plane_with_interpolation_core(**d)
            args.append(d)
        #     break
        # break
        with Pool(block_process_num) as pool:
            pool.map(thread_wrapper_plot_ab_plane_with_interpolation, args)


def thread_wrapper_plot_ab_plane_with_interpolation(args):
    plot_ab_plane_with_interpolation_core(**args)


def plot_ab_plane_with_interpolation_core(
        bg_lut_name, j_idx, j_val, color_space_name, maximum_luminance):
    if maximum_luminance <= 101:
        ab_max = 0.20
    elif maximum_luminance <= 1001:
        ab_max = 0.30
    else:
        ab_max = 0.40
    ab_sample = 1536
    hue_sample = 1536
    bg_lut = TyLchLut(np.load(bg_lut_name))
    rgb_st2084 = create_valid_jzazbz_ab_plane_image_st2084(
        j_val=j_val, ab_max=ab_max, ab_sample=ab_sample,
        color_space_name=color_space_name,
        bg_rgb_luminance=np.array([100, 100, 100]),
        maximum_luminance=maximum_luminance)

    hh_base = np.linspace(0, 360, hue_sample)
    jj_base = np.ones_like(hh_base) * j_val
    jh_array = tstack([jj_base, hh_base])

    jzczhz = bg_lut.interpolate(lh_array=jh_array)
    chroma = jzczhz[..., 1]
    hue = np.deg2rad(jzczhz[..., 2])
    aa = chroma * np.cos(hue)
    bb = chroma * np.sin(hue)

    jzazbz_luminance = jzczhz_to_jzazbz(jzczhz[0])
    xyz_lumi = jzazbz_to_large_xyz(jzazbz_luminance)
    rgb_lumi = XYZ_to_RGB(
        xyz_lumi/100, cs.D65, cs.D65,
        RGB_COLOURSPACES[cs.BT709].matrix_XYZ_to_RGB)
    # print(f"jzczhz={jzczhz[0]}")
    # print(f"xyz={xyz_lumi}")
    # print(f"rgb={rgb_lumi}")
    # lut_b = np.load(
    #     make_jzazbz_gb_lut_fname_methodb_b(
    #         color_space_name=color_space_name,
    #         luminance=maximum_luminance,
    #         lightness_num=256, hue_num=256))
    # print(f"lut={lut_b[0]}")
    # print(f"lut_c={bg_lut.lut[-1]}")

    large_xyz = jzazbz_to_large_xyz(jzazbz_luminance)
    luminance = large_xyz[1]
    graph_title = f"azbz plane,  {color_space_name},  "
    graph_title += f"Jz={j_val:.2f},  Luminance={luminance:.2f} nits"
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 12),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=graph_title,
        graph_title_size=None,
        xlabel="az", ylabel="bz",
        axis_label_size=None,
        legend_size=17,
        xlim=[-ab_max, ab_max],
        ylim=[-ab_max, ab_max],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=1,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        rgb_st2084, extent=(-ab_max, ab_max, -ab_max, ab_max), aspect='auto')
    ax1.plot(aa, bb, color='k')
    fname = "/work/overuse/2021/15_2_pass_gamut_boundary/img_seq_azbz/"
    fname += f"azbz_w_lut_{color_space_name}_"
    fname += f"{maximum_luminance}nits_{j_idx:04d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)


def thread_wrapper_plot_cj_plane_with_interpolation(args):
    plot_cj_plane_with_interpolation_core(**args)


def plot_cj_plane_with_interpolation_core(
        bg_lut_name, h_idx, h_val, color_space_name, maximum_luminance):
    bg_lut = TyLchLut(lut=np.load(bg_lut_name))
    sample_num = 1536
    jj_sample = 1536
    if maximum_luminance <= 101:
        if color_space_name == cs.BT709:
            cc_max = 0.17
            jj_max = 0.18
        elif color_space_name == cs.P3_D65:
            cc_max = 0.18
            jj_max = 0.18
        else:
            cc_max = 0.22
            jj_max = 0.18
    elif maximum_luminance <= 1001:
        if color_space_name == cs.BT709:
            cc_max = 0.27
            jj_max = 0.43
        elif color_space_name == cs.P3_D65:
            cc_max = 0.30
            jj_max = 0.43
        else:
            cc_max = 0.37
            jj_max = 0.43
    else:
        if color_space_name == cs.BT709:
            cc_max = 0.35
            jj_max = 1.0
        elif color_space_name == cs.P3_D65:
            cc_max = 0.36
            jj_max = 1.0
        else:
            cc_max = 0.45
            jj_max = 1.0
    print(f"h_val={h_val} started")

    rgb_area_img = create_valid_jzazbz_cj_plane_image_st2084(
        h_val=h_val, c_max=cc_max, l_max=jj_max,
        c_sample=sample_num, j_sample=sample_num,
        color_space_name=color_space_name,
        bg_rgb_luminance=np.array([20, 20, 20]),
        maximum_luminance=maximum_luminance)
    if maximum_luminance <= 100:
        # conv ST2084 to Gamma 2.4
        rgb_area_img = tf.oetf_from_luminance(
            tf.eotf_to_luminance(rgb_area_img, tf.ST2084), tf.GAMMA24)

    graph_title = f"CzJz plane,  {color_space_name},  hue={h_val:.2f}°,  "
    graph_title += f"target={maximum_luminance} nits"

    jj_base = np.linspace(0, bg_lut.ll_max, jj_sample)
    hh_base = np.ones_like(jj_base) * h_val
    jh_array = tstack([jj_base, hh_base])
    # jzczhz = get_gamut_boundary_lch_from_lut(
    #     lut=bg_lut, lh_array=jh_array, lightness_max=1.0)
    jzczhz = bg_lut.interpolate(lh_array=jh_array)

    chroma = jzczhz[..., 1]
    lightness = jzczhz[..., 0]

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 12),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=graph_title,
        graph_title_size=None,
        xlabel="Cz", ylabel="Jz",
        axis_label_size=None,
        legend_size=17,
        xlim=[0, cc_max],
        ylim=[0, jj_max],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=1,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        rgb_area_img, extent=(0, cc_max, 0, jj_max), aspect='auto')
    ax1.plot(chroma, lightness, color='k')
    fname = make_fname_cj_plane_with_interpolation(
        h_idx=h_idx, color_space_name=color_space_name,
        maximum_luminance=maximum_luminance)
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)


def make_fname_cj_plane_with_interpolation(
        h_idx, color_space_name, maximum_luminance):
    fname = "/work/overuse/plot_seq/czjz_plane/"
    fname += f"CzJz_w_lut_{color_space_name}_{maximum_luminance}-nits_"
    fname += f"{h_idx:04d}.png"

    return fname


def plot_cj_plane_with_interpolation(
        hue_sample=256, lightness_sample=256, h_num_intp=256,
        color_space_name=cs.BT2020, luminance=10000):

    lut_name = make_jzazbz_gb_lut_fname_method_c(
        color_space_name=color_space_name, luminance=luminance,
        lightness_num=lightness_sample, hue_num=hue_sample)

    total_process_num = h_num_intp
    block_process_num = int(cpu_count() / 3 + 0.9)
    print(f"block_process_num {block_process_num}")
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            h_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={h_idx}")  # User
            if h_idx >= total_process_num:                         # User
                break
            d = dict(
                bg_lut_name=lut_name, h_idx=h_idx,
                h_val=h_idx/(h_num_intp-1)*360,
                color_space_name=color_space_name,
                maximum_luminance=luminance)
            # plot_cj_plane_with_interpolation_core(**d)
            args.append(d)
        #     break
        # break
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_plot_cj_plane_with_interpolation, args)


def debug_plot_jzazbz(
        hue_sample=256, lightness_sample=128,
        luminance=100, h_num_intp=100, j_num_intp=100,
        color_space_name=cs.BT709):
    debug_plot_azbz_plane_with_interpolation(
        hue_sample=hue_sample, lightness_sample=lightness_sample,
        j_num_intp=j_num_intp,
        color_space_name=color_space_name, luminance=luminance)
    plot_cj_plane_with_interpolation(
        hue_sample=hue_sample, lightness_sample=lightness_sample,
        h_num_intp=h_num_intp,
        color_space_name=color_space_name, luminance=luminance)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # debug_plot_jzazbz()
