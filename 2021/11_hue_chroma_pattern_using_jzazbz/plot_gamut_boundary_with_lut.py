# -*- coding: utf-8 -*-
"""
plot gamut boundary
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from multiprocessing import Pool, cpu_count
from colour import XYZ_to_RGB, RGB_COLOURSPACES
from colour.utilities import tstack

# import my libraries
import color_space as cs
import plot_utility as pu
import transfer_functions as tf
from create_gamut_boundary_lut_jzazbz import make_lut_fname
from jzazbz import jzazbz_to_large_xyz, jzczhz_to_jzazbz, st2084_eotf_like
from create_gamut_booundary_lut import is_out_of_gamut_rgb,\
    get_gamut_boundary_lch_from_lut, calc_cusp_specific_hue,\
    calc_l_focal_specific_hue

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def create_valid_ab_plane_image_st2084(
        j_val=0.5, ab_max=1.0, ab_sample=512, color_space_name=cs.BT2020,
        bg_rgb_luminance=np.array([50, 50, 50]), maximum_luminance=10000):
    """
    Create an image that indicates the valid area of the ab plane.

    Parameters
    ----------
    j_val : float
        A Lightness value. range is 0.0 - 1.0
    ab_max : float
        A maximum value of the a, b range.
    ab_sapmle : int
        A number of samples in the image resolution.
    color_space_name : str
        color space name for colour.RGB_COLOURSPACES
    maximum_luminance : float
        maximum luminance of the target display device.
    """
    aa_base = np.linspace(-ab_max, ab_max, ab_sample)
    bb_base = np.linspace(-ab_max, ab_max, ab_sample)
    aa = aa_base.reshape((1, ab_sample))\
        * np.ones_like(bb_base).reshape((ab_sample, 1))
    bb = bb_base.reshape((ab_sample, 1))\
        * np.ones_like(aa_base).reshape((1, ab_sample))
    jj = np.ones_like(aa) * j_val
    jzazbz = np.dstack((jj, aa, bb[::-1])).reshape((ab_sample, ab_sample, 3))
    large_xyz = jzazbz_to_large_xyz(jzazbz)
    rgb_luminance = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65,
        RGB_COLOURSPACES[color_space_name].matrix_XYZ_to_RGB)

    ng_idx = is_out_of_gamut_rgb(rgb=rgb_luminance/maximum_luminance)
    rgb_luminance[ng_idx] = bg_rgb_luminance
    rgb_st2084 = tf.oetf_from_luminance(
        np.clip(rgb_luminance, 0.0, 10000), tf.ST2084)

    return rgb_st2084


def create_valid_cj_plane_image_st2084(
        h_val=50, c_max=1, c_sample=1024, j_sample=1024,
        color_space_name=cs.BT2020, bg_rgb_luminance=np.array([50, 50, 50]),
        maximum_luminance=10000):
    """
    Create an image that indicates the valid area of the ab plane.

    Parameters
    ----------
    h_val : float
        A Hue value. range is 0.0 - 360.0
    c_max : float
        A maximum value of the chroma.
    c_sapmle : int
        A number of samples for the chroma.
    l_sample : int
        A number of samples for the lightness.
    color_space_name : str
        color space name for colour.RGB_COLOURSPACES
    bg_lightness : float
        background lightness value.
    maximum_luminance : float
        maximum luminance of the target display device.
    """
    l_max = 1

    cc_base = np.linspace(0, c_max, c_sample)
    jj_base = np.linspace(0, l_max, j_sample)
    cc = cc_base.reshape(1, c_sample)\
        * np.ones_like(jj_base).reshape(j_sample, 1)
    jj = jj_base.reshape(j_sample, 1)\
        * np.ones_like(cc_base).reshape(1, c_sample)
    hh = np.ones_like(cc) * h_val

    jczhz = np.dstack([jj[::-1], cc, hh]).reshape((j_sample, c_sample, 3))
    jzazbz = jzczhz_to_jzazbz(jczhz)
    large_xyz = jzazbz_to_large_xyz(jzazbz)
    rgb_luminance = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65,
        RGB_COLOURSPACES[color_space_name].matrix_XYZ_to_RGB)
    print(f"maximum_luminance = {maximum_luminance}")
    ng_idx = is_out_of_gamut_rgb(rgb=rgb_luminance/maximum_luminance)

    rgb_luminance[ng_idx] = bg_rgb_luminance

    rgb_st2084 = tf.oetf_from_luminance(
        np.clip(rgb_luminance, 0.0, 10000), tf.ST2084)

    return rgb_st2084


def plot_cj_plane_without_interpolation_core(
        bg_lut, h_idx, h_val, color_space_name, maximum_luminance):
    cc_max = 0.5
    jj_max = 1.0
    sample_num = 1024
    print(f"h_val={h_val}")
    rgb_st2084 = create_valid_cj_plane_image_st2084(
        h_val=h_val, c_max=0.5, c_sample=sample_num, j_sample=sample_num,
        color_space_name=color_space_name,
        bg_rgb_luminance=np.array([100, 100, 100]),
        maximum_luminance=maximum_luminance)
    graph_title = f"CzJz plane,  hue={h_val:.2f},  "
    graph_title += f"target={maximum_luminance} nits"

    jzczhz = bg_lut[:, h_idx]
    chroma = jzczhz[..., 1]
    lightness = jzczhz[..., 0]

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
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
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        rgb_st2084, extent=(0, cc_max, 0, jj_max), aspect='auto')
    ax1.plot(chroma, lightness, color='k')
    fname = "/work/overuse/2021/11_chroma_hue_jzazbz/img_seq_cj/"
    fname += f"CzJz_w_lut_{color_space_name}_{h_idx:04d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)


def plot_cj_plane_with_interpolation_core(
        bg_lut, h_idx, h_val, color_space_name, maximum_luminance):
    cc_max = 0.5
    jj_max = 1.0
    sample_num = 1024
    jj_sample = 256
    print(f"h_val={h_val}")
    rgb_st2084 = create_valid_cj_plane_image_st2084(
        h_val=h_val, c_max=0.5, c_sample=sample_num, j_sample=sample_num,
        color_space_name=color_space_name,
        bg_rgb_luminance=np.array([100, 100, 100]),
        maximum_luminance=maximum_luminance)
    graph_title = f"CzJz plane,  hue={h_val:.2f},  "
    graph_title += f"target={maximum_luminance} nits"

    jj_base = np.linspace(0, 1, jj_sample)
    hh_base = np.ones_like(jj_base) * h_val
    jh_array = tstack([jj_base, hh_base])
    jzczhz = get_gamut_boundary_lch_from_lut(
        lut=bg_lut, lh_array=jh_array, lightness_max=1.0)

    chroma = jzczhz[..., 1]
    lightness = jzczhz[..., 0]

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
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
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        rgb_st2084, extent=(0, cc_max, 0, jj_max), aspect='auto')
    ax1.plot(chroma, lightness, color='k')
    fname = "/work/overuse/2021/11_chroma_hue_jzazbz/img_seq_cj_intp/"
    fname += f"CzJz_w_lut_{color_space_name}_{h_idx:04d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)


def plot_ab_plane_without_interpolation_core(
        bg_lut, j_idx, j_val, color_space_name, maximum_luminance):
    ab_max = 0.5
    ab_sample = 1024
    rgb_st2084 = create_valid_ab_plane_image_st2084(
        j_val=j_val, ab_max=ab_max, ab_sample=ab_sample,
        color_space_name=color_space_name,
        bg_rgb_luminance=np.array([100, 100, 100]),
        maximum_luminance=maximum_luminance)
    luminance = int(
        round(st2084_eotf_like(j_val)) + 0.5)
    graph_title = f"azbz plane,  Jz={j_val:.2f},  Luminance={luminance} nits"

    jzczhz = bg_lut[j_idx]
    chroma = jzczhz[..., 1]
    hue = np.deg2rad(jzczhz[..., 2])
    aa = chroma * np.cos(hue)
    bb = chroma * np.sin(hue)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
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
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        rgb_st2084, extent=(-ab_max, ab_max, -ab_max, ab_max), aspect='auto')
    ax1.plot(aa, bb, color='k')
    fname = "/work/overuse/2021/11_chroma_hue_jzazbz/img_seq_ab/"
    fname += f"azbz_w_lut_{color_space_name}_{j_idx:04d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)


def plot_ab_plane_with_interpolation_core(
        bg_lut, j_idx, j_val, color_space_name, maximum_luminance):
    ab_max = 0.5
    ab_sample = 1024
    hue_sample = 256
    rgb_st2084 = create_valid_ab_plane_image_st2084(
        j_val=j_val, ab_max=ab_max, ab_sample=ab_sample,
        color_space_name=color_space_name,
        bg_rgb_luminance=np.array([100, 100, 100]),
        maximum_luminance=maximum_luminance)
    luminance = int(
        round(st2084_eotf_like(j_val)) + 0.5)
    graph_title = f"azbz plane,  Jz={j_val:.2f},  Luminance={luminance} nits"

    hh_base = np.linspace(0, 360, hue_sample)
    jj_base = np.ones_like(hh_base) * j_val
    jh_array = tstack([jj_base, hh_base])
    # lh_array = 

    jzczhz = get_gamut_boundary_lch_from_lut(
        lut=bg_lut, lh_array=jh_array, lightness_max=1.0)
    # jzczhz = bg_lut[j_idx]
    chroma = jzczhz[..., 1]
    hue = np.deg2rad(jzczhz[..., 2])
    aa = chroma * np.cos(hue)
    bb = chroma * np.sin(hue)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
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
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        rgb_st2084, extent=(-ab_max, ab_max, -ab_max, ab_max), aspect='auto')
    ax1.plot(aa, bb, color='k')
    fname = "/work/overuse/2021/11_chroma_hue_jzazbz/img_seq_ab_intp/"
    fname += f"azbz_w_lut_{color_space_name}_{j_idx:04d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)


def thread_wrapper_plot_ab_plane_without_interpolation(args):
    plot_ab_plane_without_interpolation_core(**args)


def thread_wrapper_plot_ab_plane_with_interpolation(args):
    plot_ab_plane_with_interpolation_core(**args)


def thread_wrapper_plot_cj_plane_without_interpolation(args):
    plot_cj_plane_without_interpolation_core(**args)


def thread_wrapper_plot_cj_plane_with_interpolation(args):
    plot_cj_plane_with_interpolation_core(**args)


def thread_wrapper_plot_cups_core(args):
    plot_cups_core(**args)


def plot_ab_plane_without_interpolation():
    color_space_name = cs.BT2020
    luminance = 10000
    hue_sample = 256
    lightness_sample = 256

    lut_name = make_lut_fname(
        color_space_name=color_space_name, luminance=luminance,
        lightness_num=lightness_sample, hue_num=hue_sample)
    lut = np.load(lut_name)

    j_num = lightness_sample

    total_process_num = j_num
    block_process_num = cpu_count() // 2
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            j_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={j_idx}")  # User
            if j_idx >= total_process_num:                         # User
                break
            d = dict(
                bg_lut=lut, j_idx=j_idx, j_val=j_idx/(j_num-1),
                color_space_name=color_space_name,
                maximum_luminance=luminance)
            # plot_ab_plane_without_interpolation_core(**d)
            args.append(d)
        #     break
        # break
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_plot_ab_plane_without_interpolation, args)


def plot_ab_plane_with_interpolation():
    color_space_name = cs.BT2020
    luminance = 10000
    hue_sample = 64
    lightness_sample = 64

    lut_name = make_lut_fname(
        color_space_name=color_space_name, luminance=luminance,
        lightness_num=lightness_sample, hue_num=hue_sample)
    lut = np.load(lut_name)

    j_num = 256

    total_process_num = j_num
    block_process_num = cpu_count() // 2
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            j_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={j_idx}")  # User
            if j_idx >= total_process_num:                         # User
                break
            d = dict(
                bg_lut=lut, j_idx=j_idx, j_val=j_idx/(j_num-1),
                color_space_name=color_space_name,
                maximum_luminance=luminance)
            # plot_ab_plane_with_interpolation_core(**d)
            args.append(d)
        #     break
        # break
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_plot_ab_plane_with_interpolation, args)


def plot_cj_plane_without_interpolation():
    color_space_name = cs.BT2020
    luminance = 10000
    hue_sample = 256
    lightness_sample = 256

    lut_name = make_lut_fname(
        color_space_name=color_space_name, luminance=luminance,
        lightness_num=lightness_sample, hue_num=hue_sample)
    lut = np.load(lut_name)

    h_num = hue_sample

    total_process_num = h_num
    block_process_num = cpu_count() // 2
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            h_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={h_idx}")  # User
            if h_idx >= total_process_num:                         # User
                break
            d = dict(
                bg_lut=lut, h_idx=h_idx, h_val=h_idx/(h_num-1)*360,
                color_space_name=color_space_name,
                maximum_luminance=luminance)
            # plot_cj_plane_without_interpolation_core(**d)
            args.append(d)
        #     break
        # break
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_plot_cj_plane_without_interpolation, args)


def plot_cj_plane_with_interpolation():
    color_space_name = cs.BT2020
    luminance = 10000
    hue_sample = 64
    lightness_sample = 64

    lut_name = make_lut_fname(
        color_space_name=color_space_name, luminance=luminance,
        lightness_num=lightness_sample, hue_num=hue_sample)
    lut = np.load(lut_name)

    h_num = 256

    total_process_num = h_num
    block_process_num = cpu_count() // 2
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            h_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={h_idx}")  # User
            if h_idx >= total_process_num:                         # User
                break
            d = dict(
                bg_lut=lut, h_idx=h_idx, h_val=h_idx/(h_num-1)*360,
                color_space_name=color_space_name,
                maximum_luminance=luminance)
            # plot_cj_plane_with_interpolation_core(**d)
            args.append(d)
        #     break
        # break
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_plot_cj_plane_with_interpolation, args)


def load_gamut_boundary_lut(
        color_space_name, lightness_sample_num, hue_sample_num,
        maximum_luminance):
    lut_name = make_lut_fname(
        color_space_name=color_space_name, luminance=maximum_luminance,
        lightness_num=lightness_sample_num, hue_num=hue_sample_num)
    lut = np.load(lut_name)

    return lut


def get_interpolated_jzczhz(lut, jj_sample, h_val):
    jj_base = np.linspace(0, 1, jj_sample)
    hh_base = np.ones_like(jj_base) * h_val
    jh_array = tstack([jj_base, hh_base])

    jzczhz = get_gamut_boundary_lch_from_lut(
        lut=lut, lh_array=jh_array, lightness_max=1.0)

    return jzczhz


def plot_cups_core(
        h_idx, h_val, maximum_luminance):
    cc_max = 0.5
    jj_max = 1.0
    sample_num = 1024
    jj_sample = 1024
    print(f"h_val={h_val}")
    rgb_st2084 = create_valid_cj_plane_image_st2084(
        h_val=h_val, c_max=0.5, c_sample=sample_num, j_sample=sample_num,
        color_space_name=cs.BT2020,
        bg_rgb_luminance=np.array([100, 100, 100]),
        maximum_luminance=maximum_luminance)
    graph_title = f"CzJz plane,  hue={h_val:.2f},  "
    graph_title += f"target={maximum_luminance} nits"

    outer_lut = load_gamut_boundary_lut(
        color_space_name=cs.BT2020,
        lightness_sample_num=sample_num, hue_sample_num=sample_num,
        maximum_luminance=maximum_luminance)
    inner_lut = load_gamut_boundary_lut(
        color_space_name=cs.BT709,
        lightness_sample_num=sample_num, hue_sample_num=sample_num,
        maximum_luminance=maximum_luminance)

    outer_jzczhz = get_interpolated_jzczhz(
        lut=outer_lut, jj_sample=jj_sample, h_val=h_val)
    inner_jzczhz = get_interpolated_jzczhz(
        lut=inner_lut, jj_sample=jj_sample, h_val=h_val)

    outer_chroma = outer_jzczhz[..., 1]
    outer_lightness = outer_jzczhz[..., 0]
    inner_chroma = inner_jzczhz[..., 1]
    inner_lightness = inner_jzczhz[..., 0]

    outer_cups = calc_cusp_specific_hue(
        lut=outer_lut, hue=h_val, lightness_max=1.0)
    inner_cups = calc_cusp_specific_hue(
        lut=inner_lut, hue=h_val, lightness_max=1.0)
    focal_point = calc_l_focal_specific_hue(
        inner_lut=inner_lut, outer_lut=outer_lut, hue=h_val,
        maximum_l_focal=1.0, minimum_l_focal=0.0, lightness_max=1.0)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
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
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        rgb_st2084, extent=(0, cc_max, 0, jj_max), aspect='auto')
    ax1.plot(
        inner_chroma, inner_lightness, color='k', label="BT.709",
        lw=1.5, alpha=0.5)
    ax1.plot(outer_chroma, outer_lightness, color='k', label="BT.2020")
    ax1.plot(
        [focal_point[..., 1], outer_cups[..., 1]],
        [focal_point[..., 0], outer_cups[..., 0]], 'k--', lw=1)
    ax1.plot(
        inner_cups[..., 1], inner_cups[..., 0], 'D', markerfacecolor="None",
        markeredgecolor='k', mew=2, ms=12, label="BT.709 Cups")
    ax1.plot(
        outer_cups[..., 1], outer_cups[..., 0], 's', markerfacecolor="None",
        markeredgecolor='k', mew=2, ms=12, label="BT.2020 Cups")
    ax1.plot(
        focal_point[..., 1], focal_point[..., 0], 'o', markerfacecolor='None',
        markeredgecolor='k', ms=8, label="Focal point?")
    fname = "/work/overuse/2021/11_chroma_hue_jzazbz/img_seq_cups/"
    fname += f"cups_{h_idx:04d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc='lower right', show=False, save_fname=fname)


def plot_cups():
    luminance = 10000
    h_num = 1024

    total_process_num = h_num
    block_process_num = int(round(cpu_count() / 1.5 + 0.5))
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            h_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={h_idx}")  # User
            if h_idx >= total_process_num:                         # User
                break
            d = dict(
                h_idx=h_idx, h_val=h_idx/(h_num-1)*360,
                maximum_luminance=luminance)
            # plot_cups_core(**d)
            args.append(d)
        #     break
        # break
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_plot_cups_core, args)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # plot_ab_plane_without_interpolation()
    # plot_cj_plane_without_interpolation()

    # plot_ab_plane_with_interpolation()
    # plot_cj_plane_with_interpolation()

    plot_cups()
