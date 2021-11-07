# -*- coding: utf-8 -*-
"""
plot gamut boundary
"""

# import standard libraries
import os
import time
import subprocess
from pathlib import Path

# import third-party libraries
import numpy as np
from multiprocessing import Pool, cpu_count, shared_memory
from colour import XYZ_to_RGB, RGB_COLOURSPACES
from colour.utilities import tstack

# import my libraries
import color_space as cs
import plot_utility as pu
import transfer_functions as tf
from jzazbz import jzazbz_to_large_xyz, jzczhz_to_jzazbz, st2084_eotf_like
from create_gamut_booundary_lut import calc_l_focal_specific_hue_jzazbz,\
    is_out_of_gamut_rgb,\
    get_gamut_boundary_lch_from_lut, calc_cusp_specific_hue,\
    calc_l_focal_specific_hue,\
    make_jzazbz_focal_lut_fname_wo_lpf,\
    make_jzazbz_gb_lut_fname, make_jzazbz_focal_lut_fname, TyLchLut, make_jzazbz_gb_lut_fname_old
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

L_SAMPLE_NUM = 1024
H_SAMPLE_NUM = 4096
COLOR_NUM = 3

# shm = shared_memory.SharedMemory(
#     create=True, size=L_SAMPLE_NUM*H_SAMPLE_NUM*3*4)
# g_buf = np.ndarray(
#     (L_SAMPLE_NUM, H_SAMPLE_NUM, 3), dtype=np.float32, buffer=shm.buf)


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
        h_val=50, c_max=1, l_max=1, c_sample=1024, j_sample=1024,
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
        bg_lut_name, h_idx, h_val, color_space_name, maximum_luminance):
    bg_lut = TyLchLut(lut=np.load(bg_lut_name))
    sample_num = 1024
    jj_sample = 1024
    if maximum_luminance <= 101:
        cc_max = 0.25
        jj_max = 0.2
    elif maximum_luminance <= 1001:
        cc_max = 0.3
        jj_max = 0.5
    else:
        cc_max = 0.5
        jj_max = 1.0
    print(f"h_val={h_val} started")

    rgb_st2084 = create_valid_cj_plane_image_st2084(
        h_val=h_val, c_max=cc_max, l_max=jj_max,
        c_sample=sample_num, j_sample=sample_num,
        color_space_name=color_space_name,
        bg_rgb_luminance=np.array([100, 100, 100]),
        maximum_luminance=maximum_luminance)
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
    fname += f"CzJz_w_lut_{color_space_name}_{maximum_luminance}-nits_"
    fname += f"{h_idx:04d}.png"
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
        bg_lut_name, j_idx, j_val, color_space_name, maximum_luminance):
    if maximum_luminance <= 101:
        ab_max = 0.25
    elif maximum_luminance <= 1001:
        ab_max = 0.30
    else:
        ab_max = 0.50
    ab_sample = 1024
    hue_sample = 1024
    bg_lut = TyLchLut(np.load(bg_lut_name))
    rgb_st2084 = create_valid_ab_plane_image_st2084(
        j_val=j_val, ab_max=ab_max, ab_sample=ab_sample,
        color_space_name=color_space_name,
        bg_rgb_luminance=np.array([100, 100, 100]),
        maximum_luminance=maximum_luminance)
    jzazbz = tstack([j_val, 0.0, 0.0])
    large_xyz = jzazbz_to_large_xyz(jzazbz)
    luminance = large_xyz[1]
    graph_title = f"azbz plane,  {color_space_name},  "
    graph_title += f"Jz={j_val:.2f},  Luminance={luminance:.2f} nits"

    hh_base = np.linspace(0, 360, hue_sample)
    jj_base = np.ones_like(hh_base) * j_val
    jh_array = tstack([jj_base, hh_base])

    jzczhz = bg_lut.interpolate(lh_array=jh_array)
    # jzczhz = get_gamut_boundary_lch_from_lut(
    #     lut=bg_lut, lh_array=jh_array, lightness_max=1.0)
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
    fname += f"azbz_w_lut_{color_space_name}_"
    fname += f"{maximum_luminance}nits_{j_idx:04d}.png"
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
    # plot_cups_core_p3(**args)


def plot_ab_plane_without_interpolation():
    color_space_name = cs.BT2020
    luminance = 10000
    hue_sample = 256
    lightness_sample = 256

    lut_name = make_jzazbz_gb_lut_fname(
        color_space_name=color_space_name, luminance=luminance,
        lightness_num=lightness_sample, hue_num=hue_sample)
    lut = np.load(lut_name)

    j_num = lightness_sample

    total_process_num = j_num
    block_process_num = cpu_count() // 3
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


def plot_ab_plane_with_interpolation(
        color_space_name=cs.BT2020, luminance=10000):
    hue_sample = H_SAMPLE_NUM
    lightness_sample = L_SAMPLE_NUM

    lut_name = make_jzazbz_gb_lut_fname(
        color_space_name=color_space_name, luminance=luminance,
        lightness_num=lightness_sample, hue_num=hue_sample)
    bg_lut = TyLchLut(np.load(lut_name))
    j_max = bg_lut.ll_max

    j_num = 721

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


def plot_cj_plane_without_interpolation():
    color_space_name = cs.BT2020
    luminance = 10000
    hue_sample = H_SAMPLE_NUM
    lightness_sample = L_SAMPLE_NUM

    lut_name = make_jzazbz_gb_lut_fname(
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


def plot_cj_plane_with_interpolation(
        color_space_name=cs.BT2020, luminance=10000, lut_type='old'):
    hue_sample = H_SAMPLE_NUM
    lightness_sample = L_SAMPLE_NUM

    if lut_type == 'old':
        lut_name = make_jzazbz_gb_lut_fname_old(
            color_space_name=color_space_name, luminance=luminance,
            lightness_num=lightness_sample, hue_num=hue_sample)
    else:
        lut_name = make_jzazbz_gb_lut_fname(
            color_space_name=color_space_name, luminance=luminance,
            lightness_num=lightness_sample, hue_num=hue_sample)
    # lut = np.load(lut_name)
    # shared_array[:] = lut
    # g_buf[:] = lut
    # print(g_buf)

    h_num = 721

    total_process_num = h_num
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
                bg_lut_name=lut_name, h_idx=h_idx, h_val=h_idx/(h_num-1)*360,
                color_space_name=color_space_name,
                maximum_luminance=luminance)
            # plot_cj_plane_with_interpolation_core(**d)
            args.append(d)
        #     break
        # break
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_plot_cj_plane_with_interpolation, args)


def load_gamut_boundary_lut_old(
        color_space_name, lightness_sample_num, hue_sample_num,
        maximum_luminance):
    lut_name = make_jzazbz_gb_lut_fname_old(
        color_space_name=color_space_name, luminance=maximum_luminance,
        lightness_num=lightness_sample_num, hue_num=hue_sample_num)
    lut = TyLchLut(np.load(lut_name))

    return lut


def load_gamut_boundary_lut_new(
        color_space_name, lightness_sample_num, hue_sample_num,
        maximum_luminance):
    lut_name = make_jzazbz_gb_lut_fname(
        color_space_name=color_space_name, luminance=maximum_luminance,
        lightness_num=lightness_sample_num, hue_num=hue_sample_num)
    lut = TyLchLut(np.load(lut_name))

    return lut


def get_interpolated_jzczhz(lut, jj_sample, h_val):
    ll_max = lut.ll_max
    jj_base = np.linspace(0, ll_max, jj_sample)
    hh_base = np.ones_like(jj_base) * h_val
    jh_array = tstack([jj_base, hh_base])

    jzczhz = get_gamut_boundary_lch_from_lut(
        lut=lut.lut, lh_array=jh_array, lightness_max=ll_max)

    return jzczhz


def plot_cups_core(
        h_idx, h_val, maximum_luminance):
    if maximum_luminance <= 101:
        cc_max = 0.3
        jj_max = 0.2
    elif maximum_luminance <= 1001:
        cc_max = 0.4
        jj_max = 0.5
    else:
        cc_max = 0.5
        jj_max = 1.0
    ll_sample_num = 1024
    hh_sample_num = 4096
    jj_sample = 4096
    print(f"h_val={h_val}")
    rgb_st2084 = create_valid_cj_plane_image_st2084(
        h_val=h_val, c_max=cc_max, l_max=jj_max,
        c_sample=1024, j_sample=1024,
        color_space_name=cs.BT2020,
        bg_rgb_luminance=np.array([100, 100, 100]),
        maximum_luminance=maximum_luminance)
    graph_title = f"CzJz plane,  hue={h_val:.2f},  "
    graph_title += f"target={maximum_luminance} nits"

    outer_lut = load_gamut_boundary_lut_old(
        color_space_name=cs.BT2020,
        lightness_sample_num=ll_sample_num, hue_sample_num=hh_sample_num,
        maximum_luminance=maximum_luminance)
    inner_lut = load_gamut_boundary_lut_new(
        color_space_name=cs.BT709,
        lightness_sample_num=ll_sample_num, hue_sample_num=hh_sample_num,
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
        lut=outer_lut.lut, hue=h_val, lightness_max=outer_lut.ll_max)
    inner_cups = calc_cusp_specific_hue(
        lut=inner_lut.lut, hue=h_val, lightness_max=inner_lut.ll_max)
    focal_point = calc_l_focal_specific_hue_jzazbz(
        inner_lut=inner_lut, outer_lut=outer_lut, hue=h_val,
        maximum_l_focal=1.0, minimum_l_focal=0.0)
    hh_idx = int(h_val / 360 * (hh_sample_num - 1) + 0.9999)

    focal_point_name = make_jzazbz_focal_lut_fname(
        luminance=maximum_luminance, lightness_num=ll_sample_num,
        hue_num=hh_sample_num, prefix="BT709-BT2020")
    focal_point = np.load(focal_point_name)

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
        [focal_point[hh_idx, 1], outer_cups[..., 1]],
        [focal_point[hh_idx, 0], outer_cups[..., 0]], 'k--', lw=1)
    ax1.plot(
        inner_cups[..., 1], inner_cups[..., 0], 'D', markerfacecolor="None",
        markeredgecolor='k', mew=2, ms=12, label="BT.709 Cups")
    ax1.plot(
        outer_cups[..., 1], outer_cups[..., 0], 's', markerfacecolor="None",
        markeredgecolor='k', mew=2, ms=12, label="BT.2020 Cups")
    ax1.plot(
        focal_point[hh_idx, 1], focal_point[hh_idx, 0], 'o',
        markerfacecolor='None',
        markeredgecolor='k', ms=8, label="Focal point?")
    fname = "/work/overuse/2021/11_chroma_hue_jzazbz/img_seq_cups/"
    fname += f"cups_{maximum_luminance}-nits_{h_idx:04d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc='lower right', show=False, save_fname=fname)


def plot_cups_core_p3(
        h_idx, h_val, maximum_luminance):
    if maximum_luminance <= 101:
        cc_max = 0.3
        jj_max = 0.2
    elif maximum_luminance <= 1001:
        cc_max = 0.3
        jj_max = 0.5
    else:
        cc_max = 0.4
        jj_max = 1.0
    ll_sample_num = 1024
    hh_sample_num = 4096
    jj_sample = 4096
    print(f"h_val={h_val}")
    rgb_st2084 = create_valid_cj_plane_image_st2084(
        h_val=h_val, c_max=cc_max, l_max=jj_max,
        c_sample=1024, j_sample=1024,
        color_space_name=cs.P3_D65,
        bg_rgb_luminance=np.array([100, 100, 100]),
        maximum_luminance=maximum_luminance)
    graph_title = f"CzJz plane,  hue={h_val:.2f},  "
    graph_title += f"target={maximum_luminance} nits"

    outer_lut = load_gamut_boundary_lut_old(
        color_space_name=cs.P3_D65,
        lightness_sample_num=ll_sample_num, hue_sample_num=hh_sample_num,
        maximum_luminance=maximum_luminance)
    inner_lut = load_gamut_boundary_lut_new(
        color_space_name=cs.BT709,
        lightness_sample_num=ll_sample_num, hue_sample_num=hh_sample_num,
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
        lut=outer_lut.lut, hue=h_val, lightness_max=outer_lut.ll_max)
    inner_cups = calc_cusp_specific_hue(
        lut=inner_lut.lut, hue=h_val, lightness_max=inner_lut.ll_max)
    focal_point = calc_l_focal_specific_hue_jzazbz(
        inner_lut=inner_lut, outer_lut=outer_lut, hue=h_val,
        maximum_l_focal=1.0, minimum_l_focal=0.0)
    hh_idx = int(h_val / 360 * (hh_sample_num - 1) + 0.9999)

    focal_point_name = make_jzazbz_focal_lut_fname(
        luminance=maximum_luminance, lightness_num=ll_sample_num,
        hue_num=hh_sample_num, prefix="BT709-P3D65")
    focal_point = np.load(focal_point_name)

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
    ax1.plot(outer_chroma, outer_lightness, color='k', label="P3D65")
    ax1.plot(
        [focal_point[hh_idx, 1], outer_cups[..., 1]],
        [focal_point[hh_idx, 0], outer_cups[..., 0]], 'k--', lw=1)
    ax1.plot(
        inner_cups[..., 1], inner_cups[..., 0], 'D', markerfacecolor="None",
        markeredgecolor='k', mew=2, ms=12, label="BT.709 Cups")
    ax1.plot(
        outer_cups[..., 1], outer_cups[..., 0], 's', markerfacecolor="None",
        markeredgecolor='k', mew=2, ms=12, label="P3D65 Cups")
    ax1.plot(
        focal_point[hh_idx, 1], focal_point[hh_idx, 0], 'o',
        markerfacecolor='None',
        markeredgecolor='k', ms=8, label="Focal point?")
    fname = "/work/overuse/2021/11_chroma_hue_jzazbz/img_seq_cups/"
    fname += f"cups_P3D65_{maximum_luminance}-nits_{h_idx:04d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc='lower right', show=False, save_fname=fname)


def plot_cups(luminance=10000):
    h_num = 721

    total_process_num = h_num
    block_process_num = int(round(cpu_count() / 1.2 + 0.5))
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


def plot_focal_lut(
        luminance, lightness_num, hue_num, prefix="BT709-BT2020"):
    lut_w_lpf_name = make_jzazbz_focal_lut_fname(
        luminance=luminance, lightness_num=lightness_num,
        hue_num=hue_num, prefix=prefix)
    lut_wo_lpf_name = make_jzazbz_focal_lut_fname_wo_lpf(
        luminance=luminance, lightness_num=lightness_num,
        hue_num=hue_num, prefix=prefix)
    lut_w_lpf = np.load(lut_w_lpf_name)
    lut_wo_lpf = np.load(lut_wo_lpf_name)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=f"Focal Point,  {prefix},   {luminance}-nits",
        graph_title_size=None,
        xlabel="hz", ylabel="Jz",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=[x * 30 for x in range(13)],
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(
        lut_wo_lpf[..., 2], lut_wo_lpf[..., 0], label="without LPF",
        color=pu.RED, lw=5, alpha=0.7)
    ax1.plot(
        lut_w_lpf[..., 2], lut_w_lpf[..., 0], label="with LPF",
        color='k', lw=2)
    fname = f"./img/focal_sample_{prefix}_{luminance}-nits.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc='upper right', show=False, save_fname=fname)


def plot_rgb_near_cups():
    cups = [0.30474347, 0.23454191, 0]
    focal = 0.16838773675997634
    cc_max = cups[1]

    aa = (cups[0] - focal) / cc_max
    bb = focal

    xx = np.linspace(0, cc_max * 1.2, 1024)
    # xx = np.linspace(0.2, 0.25, 1024)
    yy = aa * xx + bb

    jz = yy
    cz = xx
    hz = np.ones_like(xx) * 0.0

    jzczhz = tstack([jz, cz, hz])
    jzazbz = jzczhz_to_jzazbz(jzczhz)
    xyz = jzazbz_to_large_xyz(jzazbz) / 1000
    rgb = XYZ_to_RGB(
        xyz, cs.D65, cs.D65,
        RGB_COLOURSPACES[cs.BT2020].matrix_XYZ_to_RGB)

    rgb_lumi = rgb * 1000
    st2084 = tf.oetf_from_luminance(
        np.clip(rgb_lumi, 0.0, 1000), tf.ST2084)

    xyz_min = np.min(xyz)
    xyz_max = np.max(xyz)
    rgb_min = np.min(rgb)
    rgb_max = np.max(rgb)
    print(rgb_min, rgb_max, cc_max)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="BT.2020, JzCzHz to RGB",
        graph_title_size=None,
        xlabel="Chroma",
        ylabel="RGB",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    # ax1.plot(xx, xyz[..., 0], "--", color=pu.RED, lw=2, label='X')
    # ax1.plot(xx, xyz[..., 1], "--", color=pu.GREEN, lw=2, label='Y')
    # ax1.plot(xx, xyz[..., 2], "--", color=pu.BLUE, lw=2, label='Z')
    ax1.plot(xx, rgb[..., 0], color=pu.RED, label='R Linear')
    ax1.plot(xx, rgb[..., 1], color=pu.GREEN, label='G Linear')
    ax1.plot(xx, rgb[..., 2], color=pu.BLUE, label='B Linear')
    ax1.plot(xx, st2084[..., 0], "--", color=pu.RED, lw=2, label='R ST2084')
    ax1.plot(xx, st2084[..., 1], "--", color=pu.GREEN, lw=2, label='G ST2084')
    ax1.plot(xx, st2084[..., 2], "--", color=pu.BLUE, lw=2, label='B ST2084')
    ax1.plot(
        [cc_max, cc_max], [rgb_min, rgb_max], 'k--', label="Boundary", lw=1.5)
    pu.show_and_save(
        fig=fig, show=False,
        legend_loc='upper left', save_fname="./img/st2084.png")


def plot_100_1000_10000_hc_plane():
    jj_sample = 1024
    cc_max = 0.5
    jj_max = 1.0

    graph_title = "CzJz plane,  BT.2020,  hue=0.0°"

    bg_lut_name_10000 = make_jzazbz_gb_lut_fname_old(
        color_space_name=cs.BT2020, luminance=10000,
        lightness_num=1024, hue_num=4096)
    gb_lut_10000 = TyLchLut(lut=np.load(bg_lut_name_10000))
    jj_base = np.linspace(0, gb_lut_10000.ll_max, jj_sample)
    hh_base = np.ones_like(jj_base) * 0.0
    jh_array = tstack([jj_base, hh_base])
    jzczhz_10000 = gb_lut_10000.interpolate(lh_array=jh_array)
    chroma_10000 = jzczhz_10000[..., 1]
    lightness_10000 = jzczhz_10000[..., 0]

    bg_lut_name_1000 = make_jzazbz_gb_lut_fname_old(
        color_space_name=cs.BT2020, luminance=1000,
        lightness_num=1024, hue_num=4096)
    gb_lut_1000 = TyLchLut(lut=np.load(bg_lut_name_1000))
    jj_base = np.linspace(0, gb_lut_1000.ll_max, jj_sample)
    hh_base = np.ones_like(jj_base) * 0.0
    jh_array = tstack([jj_base, hh_base])
    jzczhz_1000 = gb_lut_1000.interpolate(lh_array=jh_array)
    chroma_1000 = jzczhz_1000[..., 1]
    lightness_1000 = jzczhz_1000[..., 0]

    bg_lut_name_100 = make_jzazbz_gb_lut_fname_old(
        color_space_name=cs.BT2020, luminance=100,
        lightness_num=1024, hue_num=4096)
    gb_lut_100 = TyLchLut(lut=np.load(bg_lut_name_100))
    jj_base = np.linspace(0, gb_lut_100.ll_max, jj_sample)
    hh_base = np.ones_like(jj_base) * 0.0
    jh_array = tstack([jj_base, hh_base])
    jzczhz_100 = gb_lut_100.interpolate(lh_array=jh_array)
    chroma_100 = jzczhz_100[..., 1]
    lightness_100 = jzczhz_100[..., 0]

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
    ax1.plot(chroma_10000, lightness_10000, 'k-', label="10000 nits")
    ax1.plot(
        chroma_1000, lightness_1000, '--', color=pu.RED, label="1000 nits")
    ax1.plot(
        chroma_100, lightness_100, '-.', color=pu.BLUE, label="100 nits")
    ax1.plot(0.1, 0.1, 'ko', ms=12, label="P0: (Cz, Jz) = (0.1, 0.1)")
    ax1.plot(0.2, 0.3, 'kD', ms=10, label="P1: (Cz, Jz) = (0.2, 0.3)")
    ax1.plot(0.25, 0.6, 'kv', ms=12, label="P2: (Cz, Jz) = (0.25, 0.6)")
    ax1.plot(0.3, 0.8, 'ks', ms=10, label="P3: (Cz, Jz) = (0.3, 0.8)")
    fname = "./img/100_1000_10000.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc='lower right', show=False, save_fname=fname)


def save_as_csv(jch, jzazbz, xyz, rgb_10000, rgb_1000, rgb_100):
    with open("./out.csv", 'w') as f:
        buf = "R,G,B\n"
        f.write(buf)

        ndarray_list = [jch, jzazbz, xyz, rgb_10000, rgb_1000, rgb_100]
        for ndarray in ndarray_list:
            buf = ""
            for line in ndarray:
                buf += f"{line[0]},{line[1]},{line[2]},"
            f.write(buf + "\n")


def calc_blog_param():
    jch = np.array(
        [[0.1, 0.1, 0.0], [0.3, 0.2, 0.0], [0.6, 0.25, 0.0], [0.8, 0.3, 0.0]])
    jzazbz = jzczhz_to_jzazbz(jzczhz=jch)
    xyz = jzazbz_to_large_xyz(jzazbz)
    rgb = XYZ_to_RGB(
        xyz, cs.D65, cs.D65,
        RGB_COLOURSPACES[cs.BT2020].matrix_XYZ_to_RGB)
    rgb_10000 = rgb / 10000
    rgb_1000 = rgb / 1000
    rgb_100 = rgb / 100
    save_as_csv(jch, jzazbz, xyz, rgb_10000, rgb_1000, rgb_100)


def plot_bt2020_bt709_tp():
    # Hue-Chroma Pattern

    # lightness_num = L_SAMPLE_NUM
    # hue_num = H_SAMPLE_NUM
    # prefix = "BT709-BT2020"
    # luminance_list = [1000, 10000]
    # resolution_list = [[1920, 1080], [3840, 2160]]
    # hue_num_list = [32, 48, 64]
    # oetf = tf.ST2084

    # for luminance in luminance_list:
    #     for resolution in resolution_list:
    #         for tp_hue_num in hue_num_list:
    #             width = resolution[0]
    #             height = resolution[1]

    #             focal_lut_name = make_jzazbz_focal_lut_fname(
    #                 luminance=luminance, lightness_num=lightness_num,
    #                 hue_num=hue_num, prefix=prefix)
    #             focal_lut = np.load(focal_lut_name)
    #             outer_lut = load_gamut_boundary_lut_old(
    #                 color_space_name=cs.BT2020,
    #                 lightness_sample_num=lightness_num,
    #                 hue_sample_num=hue_num, maximum_luminance=luminance)
    #             inner_lut = load_gamut_boundary_lut_new(
    #                 color_space_name=cs.BT709,
    #                 lightness_sample_num=lightness_num,
    #                 hue_sample_num=hue_num, maximum_luminance=luminance)
    #             img = tpg.make_bt2020_bt709_hue_chroma_pattern_jzazbz(
    #                 focal_lut=focal_lut, outer_lut=outer_lut,
    #                 hue_num=tp_hue_num, width=width, height=height,
    #                 luminance=luminance, oetf=oetf)
    #             tp_fname = f"./img/BT2020-BT709_HC_Pattern_{width}x{height}_"
    #             tp_fname += f"h-{tp_hue_num}_{luminance}nits.png"
    #             tpg.img_wirte_float_as_16bit_int(
    #                 filename=tp_fname, img_float=img)

    luminance = 100
    resolution_list = [[1920, 1080], [3840, 2160]]
    hue_num_list = [32, 48, 64]
    lightness_num = L_SAMPLE_NUM
    hue_num = H_SAMPLE_NUM
    prefix = "BT709-BT2020"
    oetf = tf.GAMMA24

    for resolution in resolution_list:
        for tp_hue_num in hue_num_list:
            width = resolution[0]
            height = resolution[1]

            focal_lut_name = make_jzazbz_focal_lut_fname(
                luminance=luminance, lightness_num=lightness_num,
                hue_num=hue_num, prefix=prefix)
            focal_lut = np.load(focal_lut_name)
            outer_lut = load_gamut_boundary_lut_old(
                color_space_name=cs.BT2020, lightness_sample_num=lightness_num,
                hue_sample_num=hue_num, maximum_luminance=luminance)
            inner_lut = load_gamut_boundary_lut_new(
                color_space_name=cs.BT709, lightness_sample_num=lightness_num,
                hue_sample_num=hue_num, maximum_luminance=luminance)
            img = tpg.make_bt2020_bt709_hue_chroma_pattern_jzazbz(
                focal_lut=focal_lut, outer_lut=outer_lut, hue_num=tp_hue_num,
                width=width, height=height, luminance=luminance, oetf=oetf)
            tp_fname = f"./img/BT2020-BT709_HC_Pattern_{width}x{height}_"
            tp_fname += f"h-{tp_hue_num}_{luminance}nits.png"
            tpg.img_wirte_float_as_16bit_int(filename=tp_fname, img_float=img)

            icc_profile =\
                './icc_profile/Gamma2.4_BT.2020_D65.icc'
            fname_with_profile =\
                "./img/" + Path(tp_fname).stem + "_icc_profile.png"
            cmd = [
                'convert', tp_fname, '-profile', icc_profile,
                fname_with_profile]
            subprocess.run(cmd)

            img = tpg.make_bt709_hue_chroma_pattern_jzazbz(
                focal_lut=focal_lut, outer_lut=inner_lut, hue_num=tp_hue_num,
                width=width, height=height, luminance=luminance)
            tp_fname = f"./img/BT709_HC_Pattern_{width}x{height}_h-{hue_num}_"
            tp_fname += f"{luminance}nits.png"
            tpg.img_wirte_float_as_16bit_int(filename=tp_fname, img_float=img)

            icc_profile =\
                './icc_profile/Gamma2.4_BT.709_D65.icc'
            fname_with_profile =\
                "./img/" + Path(tp_fname).stem + "_icc_profile.png"
            cmd = [
                'convert', tp_fname, '-profile', icc_profile,
                fname_with_profile]
            subprocess.run(cmd)


def plot_p3d65_bt709_tp():
    # Hue-Chroma Pattern

    # lightness_num = L_SAMPLE_NUM
    # hue_num = H_SAMPLE_NUM
    # prefix = "BT709-P3D65"
    # luminance_list = [1000, 10000]
    # resolution_list = [[1920, 1080], [3840, 2160]]
    # hue_num_list = [32, 48, 64]
    # oetf = tf.ST2084

    # for luminance in luminance_list:
    #     for resolution in resolution_list:
    #         for tp_hue_num in hue_num_list:
    #             width = resolution[0]
    #             height = resolution[1]

    #             focal_lut_name = make_jzazbz_focal_lut_fname(
    #                 luminance=luminance, lightness_num=lightness_num,
    #                 hue_num=hue_num, prefix=prefix)
    #             focal_lut = np.load(focal_lut_name)
    #             outer_lut = load_gamut_boundary_lut_old(
    #                 color_space_name=cs.P3_D65,
    #                 lightness_sample_num=lightness_num,
    #                 hue_sample_num=hue_num, maximum_luminance=luminance)
    #             img = tpg.make_p3d65_bt709_hue_chroma_pattern_jzazbz(
    #                 focal_lut=focal_lut, outer_lut=outer_lut,
    #                 hue_num=tp_hue_num, width=width, height=height,
    #                 luminance=luminance, oetf=oetf)
    #             tp_fname = f"./img/P3D65-BT709_HC_Pattern_{width}x{height}_"
    #             tp_fname += f"h-{tp_hue_num}_{luminance}nits.png"
    #             tpg.img_wirte_float_as_16bit_int(
    #                 filename=tp_fname, img_float=img)

    luminance = 100
    resolution_list = [[1920, 1080], [3840, 2160]]
    hue_num_list = [32, 48, 64]
    lightness_num = L_SAMPLE_NUM
    hue_num = H_SAMPLE_NUM
    prefix = "BT709-P3D65"
    oetf = tf.GAMMA24

    for resolution in resolution_list:
        for tp_hue_num in hue_num_list:
            width = resolution[0]
            height = resolution[1]

            focal_lut_name = make_jzazbz_focal_lut_fname(
                luminance=luminance, lightness_num=lightness_num,
                hue_num=hue_num, prefix=prefix)
            focal_lut = np.load(focal_lut_name)
            outer_lut = load_gamut_boundary_lut_old(
                color_space_name=cs.P3_D65, lightness_sample_num=lightness_num,
                hue_sample_num=hue_num, maximum_luminance=luminance)
            inner_lut = load_gamut_boundary_lut_new(
                color_space_name=cs.BT709, lightness_sample_num=lightness_num,
                hue_sample_num=hue_num, maximum_luminance=luminance)
            img = tpg.make_p3d65_bt709_hue_chroma_pattern_jzazbz(
                focal_lut=focal_lut, outer_lut=outer_lut, hue_num=tp_hue_num,
                width=width, height=height, luminance=luminance, oetf=oetf)
            tp_fname = f"./img/P3D65-BT709_HC_Pattern_{width}x{height}_"
            tp_fname += f"h-{tp_hue_num}_{luminance}nits.png"
            tpg.img_wirte_float_as_16bit_int(filename=tp_fname, img_float=img)

            icc_profile =\
                './icc_profile/Gamma2.4_DCI-P3_D65.icc'
            fname_with_profile =\
                "./img/" + Path(tp_fname).stem + "_icc_profile.png"
            cmd = [
                'convert', tp_fname, '-profile', icc_profile,
                fname_with_profile]
            subprocess.run(cmd)

            img = tpg.make_bt709_hue_chroma_pattern_jzazbz(
                focal_lut=focal_lut, outer_lut=inner_lut, hue_num=tp_hue_num,
                width=width, height=height, luminance=luminance)
            tp_fname = f"./img/BT709_HC_Pattern_{width}x{height}_h-{hue_num}_"
            tp_fname += f"{luminance}nits.png"
            tpg.img_wirte_float_as_16bit_int(filename=tp_fname, img_float=img)

            icc_profile =\
                './icc_profile/Gamma2.4_BT.709_D65.icc'
            fname_with_profile =\
                "./img/" + Path(tp_fname).stem + "_icc_profile.png"
            cmd = [
                'convert', tp_fname, '-profile', icc_profile,
                fname_with_profile]
            subprocess.run(cmd)


def plot_focal_lut_all():
    plot_focal_lut(
        luminance=10000, lightness_num=1024, hue_num=4096,
        prefix="BT709-BT2020")
    plot_focal_lut(
        luminance=1000, lightness_num=1024, hue_num=4096,
        prefix="BT709-BT2020")
    plot_focal_lut(
        luminance=100, lightness_num=1024, hue_num=4096,
        prefix="BT709-BT2020")

    plot_focal_lut(
        luminance=10000, lightness_num=1024, hue_num=4096,
        prefix="BT709-P3D65")
    plot_focal_lut(
        luminance=1000, lightness_num=1024, hue_num=4096,
        prefix="BT709-P3D65")
    plot_focal_lut(
        luminance=100, lightness_num=1024, hue_num=4096,
        prefix="BT709-P3D65")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # plot_focal_lut_all()
    # plot_bt2020_bt709_tp()
    plot_p3d65_bt709_tp()

    # plot_ab_plane_without_interpolation()
    # plot_cj_plane_without_interpolation()

    # plot_ab_plane_with_interpolation(
    #     color_space_name=cs.BT709, luminance=100)
    # plot_ab_plane_with_interpolation(
    #     color_space_name=cs.BT2020, luminance=100)
    # plot_cj_plane_with_interpolation(
    #     color_space_name=cs.BT709, luminance=10000)
    # plot_cj_plane_with_interpolation(
    #     color_space_name=cs.BT2020, luminance=10000)
    # plot_cj_plane_with_interpolation(
    #     color_space_name=cs.BT709, luminance=1000)
    # plot_cj_plane_with_interpolation(
    #     color_space_name=cs.BT2020, luminance=1000)
    # plot_cj_plane_with_interpolation(
    #     color_space_name=cs.BT709, luminance=100)
    # plot_cj_plane_with_interpolation(
    #     color_space_name=cs.BT2020, luminance=100)

    # plot_cj_plane_with_interpolation(
    #     color_space_name=cs.P3_D65, luminance=10000)
    # plot_cj_plane_with_interpolation(
    #     color_space_name=cs.P3_D65, luminance=1000)
    # plot_cj_plane_with_interpolation(
    #     color_space_name=cs.P3_D65, luminance=100)

    plot_cups(luminance=10000)
    plot_cups(luminance=1000)
    plot_cups(luminance=100)

    # plot_rgb_near_cups()

    # plot_100_1000_10000_hc_plane()
    # calc_blog_param()
