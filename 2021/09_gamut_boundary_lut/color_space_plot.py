# -*- coding: utf-8 -*-
"""
create gamut boundary lut.
"""

# import standard libraries
import os
from multiprocessing import Pool, cpu_count
import subprocess
from pathlib import Path

# import third-party libraries
import numpy as np
from colour import LCHab_to_Lab
from colour.utilities.array import tstack

# import my libraries
import create_gamut_booundary_lut as cgb
import color_space as cs
import transfer_functions as tf
import test_pattern_generator2 as tpg
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def lab_to_rgb_srgb(lab, color_space_name):
    rgb_linear = cs.lab_to_rgb(lab=lab, color_space_name=color_space_name)
    srgb = tf.oetf(np.clip(rgb_linear, 0.0, 1.0), tf.SRGB)

    return srgb


def debug_plot_cl_plane_core(hue, color_space_name=cs.BT2020):
    """
    Parameters
    ----------
    hue : float
        hue. the unit is degree. range is 0.0 - 360.0
    color_space_name : str
        color space name
    """
    c_sample = 1280 * 2
    l_sample = 720 * 2
    c_max = 220

    srgb = create_valid_cl_plane_image_srgb(
        h_val=hue, c_max=c_max, c_sample=c_sample, l_sample=l_sample,
        color_space_name=color_space_name, bg_val=0.5)
    fname = f"./img/{color_space_name}_h-{hue:.1f}_"
    fname += f"c-{c_sample}_l_{l_sample}.png"
    tpg.img_write(fname, np.uint8(srgb * 255), comp_val=6)


def create_valid_ab_plane_image_srgb(
        l_val=50, ab_max=200, ab_sample=512, color_space_name=cs.BT2020,
        bg_rgb=np.array([0.5, 0.5, 0.5])):
    """
    Create an image that indicates the valid area of the ab plane.

    Parameters
    ----------
    l_val : float
        A Lightness value. range is 0.0 - 100.0
    ab_max : float
        A maximum value of the a, b range.
    ab_sapmle : int
        A number of samples in the image resolution.
    color_space_name : str
        color space name for colour.RGB_COLOURSPACES
    """
    dummy_rgb = bg_rgb
    aa_base = np.linspace(-ab_max, ab_max, ab_sample)
    bb_base = np.linspace(-ab_max, ab_max, ab_sample)
    aa = aa_base.reshape((1, ab_sample))\
        * np.ones_like(bb_base).reshape((ab_sample, 1))
    bb = bb_base.reshape((ab_sample, 1))\
        * np.ones_like(aa_base).reshape((1, ab_sample))
    ll = np.ones_like(aa) * l_val
    lab = np.dstack((ll, aa, bb[::-1])).reshape((ab_sample, ab_sample, 3))
    rgb = cs.lab_to_rgb(lab=lab, color_space_name=color_space_name)
    ng_idx = cgb.is_out_of_gamut_rgb(rgb=rgb)
    rgb[ng_idx] = dummy_rgb
    srgb_image = tf.oetf(np.clip(rgb, 0.0, 1.0), tf.SRGB)

    return srgb_image


def create_valid_cl_plane_image_srgb(
        h_val=50, c_max=220, c_sample=1280, l_sample=720,
        color_space_name=cs.BT2020, bg_val=0.5):
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
    """
    l_max = 100
    dummy_rgb = np.array([bg_val, bg_val, bg_val])

    cc_base = np.linspace(0, c_max, c_sample)
    ll_base = np.linspace(0, l_max, l_sample)
    cc = cc_base.reshape(1, c_sample)\
        * np.ones_like(ll_base).reshape(l_sample, 1)
    ll = ll_base.reshape(l_sample, 1)\
        * np.ones_like(cc_base).reshape(1, c_sample)
    hh = np.ones_like(cc) * h_val

    lch = np.dstack([ll[::-1], cc, hh]).reshape((l_sample, c_sample, 3))
    lab = LCHab_to_Lab(lch)
    rgb = cs.lab_to_rgb(lab=lab, color_space_name=color_space_name)
    ng_idx = cgb.is_out_of_gamut_rgb(rgb=rgb)
    # ng_idx = cgb.is_outer_gamut(lab=lab, color_space_name=color_space_name)
    rgb[ng_idx] = dummy_rgb

    srgb = tf.oetf(np.clip(rgb, 0.0, 1.0), tf.SRGB)

    return srgb


def debug_plot_ab_plane(l_val, color_space_name, idx=0):
    ab_max = 200
    ab_sample = 2560

    srgb_image = create_valid_ab_plane_image_srgb(
        ab_max=ab_max, ab_sample=ab_sample, color_space_name=color_space_name)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(20, 16),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=f"ab plane, L*={l_val:.2f}",
        graph_title_size=None,
        xlabel="a*", ylabel="b*",
        axis_label_size=None,
        legend_size=17,
        xlim=[-200, 200],
        ylim=[-200, 200],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot([0], [0], '.')
    ax1.imshow(
        srgb_image, extent=(-ab_max, ab_max, -ab_max, ab_max), aspect='auto')
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False,
        save_fname=f"./img/test_ab_{idx:04d}.png")


def plot_ab_plane_core(ty_ch_lut, l_idx, l_val, color_space_name):
    """
    Parameters
    ----------
    ty_ch_lut : ndarray
        gamut boundary data.
        shape is (Hue_num, 3).
        the data order is L*, C*, Hab
    l_idx : int
        A Lightness index for ty_ch_lut
    l_val : float
        Lightness Value of the ab plane
    color_space_name : str
        color space name for colour.RGB_COLOURSPACES
    """
    lch = ty_ch_lut[l_idx]
    lab = LCHab_to_Lab(lch)

    if l_val < 25:
        line_color = (0.96, 0.96, 0.96)
    else:
        line_color = (0.005, 0.005, 0.005)

    ab_max = 200
    ab_sample = 1280

    srgb_image = create_valid_ab_plane_image_srgb(
        l_val=l_val, ab_max=ab_max, ab_sample=ab_sample,
        color_space_name=color_space_name)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(20, 16),
        bg_color=None,
        graph_title=f"ab plane, L*={l_val:.2f}",
        graph_title_size=None,
        xlabel="a*", ylabel="b*",
        axis_label_size=None,
        legend_size=17,
        xlim=[-200, 200],
        ylim=[-200, 200],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=4,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        srgb_image, extent=(-ab_max, ab_max, -ab_max, ab_max), aspect='auto')
    ax1.plot(
        lab[..., 1], lab[..., 2], '-', color=line_color,
        label="Boundaries using LUT", alpha=0.6)
    pu.show_and_save(
        fig=fig, legend_loc='upper right', show=False,
        save_fname=f"./img/ab_plane_{l_idx:04d}.png")


def plot_ab_plane_using_intp_core(
        ty_ch_lut, l_idx, l_val, color_space_name):
    """
    Parameters
    ----------
    ty_ch_lut : ndarray
        gamut boundary data.
        shape is (Hue_num, 3).
        the data order is L*, C*, Hab
    l_idx : int
        A Lightness index for ty_ch_lut
    l_val : float
        Lightness Value of the ab plane
    color_space_name : str
        color space name for colour.RGB_COLOURSPACES
    """
    hue_array = np.linspace(0, 360, 512)
    ll_array = np.ones_like(hue_array) * l_val
    lh_array = tstack([ll_array, hue_array])
    lch = cgb.get_gamut_boundary_lch_from_lut(
        lut=ty_ch_lut, lh_array=lh_array)
    lab = LCHab_to_Lab(lch)

    if l_val < 25:
        line_color = (0.96, 0.96, 0.96)
    else:
        line_color = (0.005, 0.005, 0.005)

    ab_max = 200
    ab_sample = 1280

    srgb_image = create_valid_ab_plane_image_srgb(
        l_val=l_val, ab_max=ab_max, ab_sample=ab_sample,
        color_space_name=color_space_name)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(20, 16),
        bg_color=None,
        graph_title=f"ab plane, L*={l_val:.2f}",
        graph_title_size=None,
        xlabel="a*", ylabel="b*",
        axis_label_size=None,
        legend_size=17,
        xlim=[-200, 200],
        ylim=[-200, 200],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=4,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        srgb_image, extent=(-ab_max, ab_max, -ab_max, ab_max), aspect='auto')
    ax1.plot(
        lab[..., 1], lab[..., 2], '-', color=line_color,
        label="Boundaries using LUT", alpha=0.6)
    prefix = "/work/overuse/2021/09_gamut_boundary/img_seq/"
    fname = f"{prefix}/intp_ab_plane_{l_idx:04d}.png"
    print(f"save file = {fname}")
    pu.show_and_save(
        fig=fig, legend_loc='upper right', show=False, save_fname=fname)


def plot_cl_plane_using_intp_core(
        ty_lch_lut, h_idx, h_val, color_space_name):
    """
    Parameters
    ----------
    ty_lch_lut : ndarray
        gamut boundary data.
        shape is (Lightness_num, Hue_num, 3).
        the data order is L*, C*, Hab
    h_idx : int
        A Hue index for ty_ch_lut
    h_val : float
        Hue Value of the ab plane
    color_space_name : str
        color space name for colour.RGB_COLOURSPACES
    """
    lightness_array = np.linspace(0, 100, 512)
    hh_array = np.ones_like(lightness_array) * h_val
    lh_array = tstack([lightness_array, hh_array])
    lch = cgb.get_gamut_boundary_lch_from_lut(
        lut=ty_lch_lut, lh_array=lh_array)
    # lab = LCHab_to_Lab(lch)
    line_color = (0.005, 0.005, 0.005)

    c_max = 220
    c_sample = 1280
    l_max = 100
    l_sample = 720

    srgb_image = create_valid_cl_plane_image_srgb(
        h_val=h_val, c_max=c_max, c_sample=c_sample, l_sample=l_sample,
        color_space_name=color_space_name, bg_val=0.5)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(20, 16),
        bg_color=None,
        graph_title=f"cl plane, H*={h_val:.2f}",
        graph_title_size=None,
        xlabel="Chroma", ylabel="Lightness",
        axis_label_size=None,
        legend_size=17,
        xlim=[0, c_max],
        ylim=[0, l_max],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=4,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        srgb_image, extent=(0, c_max, 0, l_max), aspect='auto')
    ax1.plot(
        lch[..., 1], lch[..., 0], '-', color=line_color,
        label="Boundaries using LUT", alpha=0.6)
    prefix = "/work/overuse/2021/09_gamut_boundary/img_seq/"
    fname = f"{prefix}/intp_cl_plane_{h_idx:04d}.png"
    print(f"save file = {fname}")
    pu.show_and_save(
        fig=fig, legend_loc='upper right', show=False, save_fname=fname)


def plot_cl_plane_using_intp_for_blog_make_lut_image(
        h_idx, h_val, color_space_name):
    """
    Parameters
    ----------
    h_idx : int
        A Hue index for ty_ch_lut
    h_val : float
        Hue Value of the ab plane
    color_space_name : str
        color space name for colour.RGB_COLOURSPACES
    """
    c_max = 220
    c_sample = 240
    l_max = 100
    l_sample = 135

    srgb_image = create_valid_cl_plane_image_srgb(
        h_val=h_val, c_max=c_max, c_sample=c_sample, l_sample=l_sample,
        color_space_name=color_space_name, bg_val=0.5)

    pp = np.uint8(np.round(tf.oetf(0.5, tf.SRGB) * 255))
    print(pp)

    img_8bit = np.uint8(np.round(srgb_image * 255))

    for l_idx in range(l_sample):
        print(l_idx)
        for c_idx in range(c_sample):
            # print((srgb_image[l_idx, c_idx] == pp).all())
            if (img_8bit[l_idx, c_idx] == pp).all():
                if c_idx == 0:
                    continue
                ok_idx = c_idx - 1 if c_idx > 0 else 0
                img_8bit[l_idx, ok_idx] = np.array([0, 0, 0])
                break

    srgb_image = img_8bit / 255

    fig, ax1 = pu.plot_1_graph(
        fontsize=14,
        figsize=(8, 5),
        bg_color=None,
        graph_title=f"BT.2020 Gamut Boundary, Hue={h_val:.2f}",
        graph_title_size=None,
        xlabel="Chroma", ylabel="Lightness",
        axis_label_size=None,
        legend_size=17,
        xlim=[0, c_max],
        ylim=[0, l_max],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=4,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        srgb_image, extent=(0, c_max, 0, l_max), aspect='auto')
    fname = f"./blog_img/gamut_boundary_{h_idx:04d}.png"
    print(f"save file = {fname}")
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)


def thread_wrapper_plot_ab_plane_using_intp_core(args):
    plot_ab_plane_using_intp_core(**args)


def thread_wrapper_plot_cl_plane_using_intp_core(args):
    plot_cl_plane_using_intp_core(**args)


def thread_wrapper_plot_ab_plane_core(args):
    plot_ab_plane_core(**args)


def thread_wrapper_plot_l_focal_and_cups_core(args):
    plot_l_focal_and_cups_core(**args)


def plot_ab_plane_seq(ty_lch_lut, color_space_name):
    """
    Parameters
    ----------
    ty_lch_lut : ndarray
        gamut boundary data.
        shape is (Lightness_num, Hue_num, 3).
        the data order is L*, C*, Hab
    """
    l_num = ty_lch_lut.shape[0]

    total_process_num = l_num
    block_process_num = cpu_count()
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            l_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={l_idx}")  # User
            if l_idx >= total_process_num:                         # User
                break
            d = dict(
                ty_ch_lut=ty_lch_lut, l_idx=l_idx,          # User
                l_val=l_idx/(l_num-1)*100, color_space_name=color_space_name)
            # plot_ab_plane_core(**d)
            args.append(d)
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_plot_ab_plane_core, args)


def plot_ab_plane_seq_using_intp(ty_lch_lut, color_space_name):
    """
    Parameters
    ----------
    ty_lch_lut : ndarray
        gamut boundary data.
        shape is (Lightness_num, Hue_num, 3).
        the data order is L*, C*, Hab
    """
    l_num = 1001

    total_process_num = l_num
    block_process_num = cpu_count()
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            l_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={l_idx}")  # User
            if l_idx >= total_process_num:                         # User
                break
            d = dict(
                ty_ch_lut=ty_lch_lut, l_idx=l_idx,          # User
                l_val=l_idx/(l_num-1)*100, color_space_name=color_space_name)
            # plot_ab_plane_using_intp_core(**d)
            args.append(d)
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_plot_ab_plane_using_intp_core, args)


def plot_cl_plane_seq_using_intp(ty_lch_lut, color_space_name):
    """
    Parameters
    ----------
    ty_lch_lut : ndarray
        gamut boundary data.
        shape is (Lightness_num, Hue_num, 3).
        the data order is L*, C*, Hab
    """
    h_num = 1001

    total_process_num = h_num
    block_process_num = cpu_count()
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            h_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={h_idx}")  # User
            if h_idx >= total_process_num:                         # User
                break
            d = dict(
                ty_ch_lut=ty_lch_lut, h_idx=h_idx,          # User
                h_val=h_idx/(h_num-1)*360, color_space_name=color_space_name)
            # plot_cl_plane_using_intp_core(**d)
            args.append(d)
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_plot_cl_plane_using_intp_core, args)


def plot_l_focal_and_cups_core(
        inner_ty_lch_lut, outer_ty_lch_lut, h_idx, h_val, color_space_name):
    lightness_array = np.linspace(0, 100, 512)
    hh_array = np.ones_like(lightness_array) * h_val
    lh_array = tstack([lightness_array, hh_array])
    inner_lch = cgb.get_gamut_boundary_lch_from_lut(
        lut=inner_ty_lch_lut, lh_array=lh_array)
    outer_lch = cgb.get_gamut_boundary_lch_from_lut(
        lut=outer_ty_lch_lut, lh_array=lh_array)
    line_color = (0.2, 0.2, 0.2)
    maximum_l_focal = 90
    minimum_l_focal = 50

    inner_cusp = cgb.calc_cusp_specific_hue(lut=inner_lut, hue=h_val)
    outer_cusp = cgb.calc_cusp_specific_hue(lut=outer_lut, hue=h_val)
    l_focal = cgb.calc_l_focal_specific_hue(
        inner_lut=inner_lut, outer_lut=outer_lut, hue=h_val,
        maximum_l_focal=maximum_l_focal, minimum_l_focal=minimum_l_focal)
    inner_color = lab_to_rgb_srgb(
        lab=LCHab_to_Lab(inner_cusp), color_space_name=cs.BT2020)
    outer_color = lab_to_rgb_srgb(
        lab=LCHab_to_Lab(outer_cusp), color_space_name=cs.BT2020)

    c_max = 220
    l_max = 104

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(16, 10),
        bg_color=(0.85, 0.85, 0.85),
        graph_title=f"L_focal, BT.709 cups, BT.2020 cups, H={h_val:.2f}",
        graph_title_size=None,
        xlabel="Chroma", ylabel="Lightness",
        axis_label_size=None,
        legend_size=17,
        xlim=[-4, c_max],
        ylim=[-4, l_max],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=4,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(
        inner_lch[..., 1], inner_lch[..., 0], '-', color=line_color,
        label="BT.709", alpha=0.7, lw=2)
    ax1.plot(
        outer_lch[..., 1], outer_lch[..., 0], '-', color=line_color,
        label="BT.2020", alpha=1.0)
    ax1.plot(
        [l_focal[1], outer_cusp[1]], [l_focal[0], outer_cusp[0]], '--', lw=1,
        color=line_color)
    ax1.plot(
        inner_cusp[1], inner_cusp[0], 's', label="BT.709 cups", ms=22,
        alpha=0.8, color=inner_color)
    ax1.plot(
        outer_cusp[1], outer_cusp[0], 's', label="BT.2020 cups", ms=22,
        alpha=0.8, color=outer_color)
    ax1.plot(
        l_focal[1], l_focal[0], 'x', label="L focal", ms=22, mew=5,
        color=line_color)
    prefix = "/work/overuse/2021/09_gamut_boundary/l_focal_max90"
    fname = f"{prefix}/L_focal_cl_plane_max-{maximum_l_focal}_{h_idx:04d}.png"
    print(f"save file = {fname}")
    pu.show_and_save(
        fig=fig, legend_loc='lower right', show=False, save_fname=fname)


def plot_l_focal_and_cups_seq(
        inner_lch_lut, outer_lch_lut, color_space_name):
    """
    Parameters
    ----------
    inner_lch_lut : ndarray
        gamut boundary data.
        shape is (Lightness_num, Hue_num, 3).
        the data order is L*, C*, Hab
    outer_lch_lut : ndarray
        gamut boundary data.
        shape is (Lightness_num, Hue_num, 3).
        the data order is L*, C*, Hab
    color_space_name : str
        color space name for colour.RGB_COLOURSPACES
    """
    h_num = 361

    total_process_num = h_num
    block_process_num = cpu_count()
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            h_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={h_idx}")  # User
            if h_idx >= total_process_num:                         # User
                break
            d = dict(
                inner_ty_lch_lut=inner_lch_lut, outer_ty_lch_lut=outer_lch_lut,
                h_idx=h_idx, h_val=h_idx/(h_num-1)*360,
                color_space_name=color_space_name)
            # plot_l_focal_and_cups_core(**d)
            args.append(d)
        #     break
        # break
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_plot_l_focal_and_cups_core, args)


def check_lch_2d_lut():
    lut = np.load("./lut/lut_sample_50_361_8192.npy")
    plot_ab_plane_core(
        ty_ch_lut=lut, l_idx=40, l_val=40/49*100, color_space_name=cs.BT2020)
    plot_ab_plane_core(
        ty_ch_lut=lut, l_idx=25, l_val=25/49*100, color_space_name=cs.BT2020)
    plot_ab_plane_core(
        ty_ch_lut=lut, l_idx=10, l_val=10/49*100, color_space_name=cs.BT2020)


def debug_hue_chroma_pattern_core(
        inner_lut, outer_lut, h_idx, h_val, chroma_num):
    lch_array = tpg._calc_l_focal_to_cups_lch_array(
        inner_lut=inner_lut, outer_lut=outer_lut,
        h_val=h_val, chroma_num=chroma_num,
        l_focal_max=90, l_focal_min=50)
    l_focal = cgb.calc_l_focal_specific_hue(
        inner_lut=inner_lut, outer_lut=outer_lut, hue=h_val,
        maximum_l_focal=90, minimum_l_focal=50)

    rgb_array = lab_to_rgb_srgb(LCHab_to_Lab(lch_array), cs.BT709)
    outer_cusp = cgb.calc_cusp_specific_hue(lut=outer_lut, hue=h_val)

    lightness_array = np.linspace(0, 100, 1024)
    hh_array = np.ones_like(lightness_array) * h_val
    lh_array = tstack([lightness_array, hh_array])
    inner_lch = cgb.get_gamut_boundary_lch_from_lut(
        lut=inner_lut, lh_array=lh_array)
    outer_lch = cgb.get_gamut_boundary_lch_from_lut(
        lut=outer_lut, lh_array=lh_array)
    line_color_1 = (0.1, 0.1, 0.1)
    line_color_2 = (0.6, 0.6, 0.6)

    c_max =150
    l_max = 102
    title = f"Chroma-Lightness Plane, Hue={h_val:.1f}°"

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 7),
        bg_color=(0.85, 0.85, 0.85),
        graph_title=title,
        graph_title_size=None,
        xlabel="Chroma", ylabel="Lightness",
        axis_label_size=None,
        legend_size=17,
        xlim=[-4, c_max],
        ylim=[90, l_max],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=5,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(
        outer_lch[..., 1], outer_lch[..., 0], '-', color=line_color_2,
        label="BT.2020", alpha=1.0)
    ax1.plot(
        inner_lch[..., 1], inner_lch[..., 0], '-', color=line_color_1,
        label="BT.709", alpha=1.0, lw=2)
    ax1.plot(
        [l_focal[1], outer_cusp[1]], [l_focal[0], outer_cusp[0]],
        'k--', lw=1)
    # ax1.scatter(
    #     lch_array[..., 1], lch_array[..., 0], zorder=3, c=rgb_array,
    #     edgecolors='k', s=400, alpha=0.7, marker='s')
    ax1.plot(
        lch_array[0, 1], lch_array[0, 0], 'x', label='Focal Point')
    prefix = "/work/overuse/2021/09_gamut_boundary/debug_hc"
    fname = f"{prefix}/not_restricted_focal_l-80_{h_idx:04d}.png"
    print(f"save file = {fname}")
    pu.show_and_save(
        fig=fig, legend_loc='lower right', show=False, save_fname=fname)


def plot_hue_l_focal_plane(inner_lut, outer_lut):
    hue = np.linspace(0, 360, 361)
    l_focal = np.zeros_like(hue)
    for h_idx, h_val in enumerate(hue):
        l_focal_lch = cgb.calc_l_focal_specific_hue(
            inner_lut=inner_lut, outer_lut=outer_lut, hue=h_val,
            maximum_l_focal=95)
        l_focal[h_idx] = l_focal_lch[0]

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 6),
        bg_color=(0.94, 0.94, 0.94),
        graph_title="L_focal",
        graph_title_size=None,
        xlabel="Hue", ylabel="L_focal",
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
    ax1.plot(hue, l_focal, label="L_focal", color=(0.2, 0.2, 0.2))
    fname = "img/l_focal_restricted.png"
    print(f"save file = {fname}")
    pu.show_and_save(
        fig=fig, legend_loc='lower right', show=True, save_fname=fname)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # lut = np.load("./lut/lut_sample_1024_1024_32768.npy")
    # plot_cl_plane_seq_using_intp(ty_lch_lut=lut, color_space_name=cs.BT2020)
    # plot_ab_plane_seq_using_intp(ty_lch_lut=lut, color_space_name=cs.BT2020)
    resolution_list = [(1920, 1080), (3840, 2160)]
    hue_num_list = [32, 48, 64, 96, 128]

    inner_lut = np.load("./lut/lut_sample_1024_1024_32768_ITU-R BT.709.npy")
    outer_lut = np.load("./lut/lut_sample_1024_1024_32768_ITU-R BT.2020.npy")
    # plot_l_focal_and_cups_seq(
    #     inner_lch_lut=inner_lut, outer_lch_lut=outer_lut,
    #     color_space_name=cs.BT2020)
    icc_profile = "./icc_profile/Gamma2.4_BT.2020_D65.icc"
    for resolution in resolution_list:
        for hue_num in hue_num_list:
            width = resolution[0]
            height = resolution[1]
            img = tpg.make_bt2020_bt709_hue_chroma_pattern(
                inner_lut=inner_lut, outer_lut=outer_lut,
                width=width, height=height, hue_num=hue_num)
            fname = f"./bt2020_bt709_hue_chroma_{width}x{height}_"
            fname += f"h_num-{hue_num}.png"
            print(fname)
            fname_with_profile = str('img' / Path(fname))
            tpg.img_wirte_float_as_16bit_int(fname, img)
            cmd = [
                'convert', fname, '-profile', icc_profile, fname_with_profile]
            subprocess.run(cmd)
            os.remove(fname)

    inner_lut = np.load("./lut/lut_sample_1024_1024_32768_P3-D65.npy")
    outer_lut = np.load("./lut/lut_sample_1024_1024_32768_ITU-R BT.2020.npy")
    for resolution in resolution_list:
        for hue_num in hue_num_list:
            width = resolution[0]
            height = resolution[1]
            img = tpg.make_bt2020_dci_p3_hue_chroma_pattern(
                inner_lut=inner_lut, outer_lut=outer_lut,
                width=resolution[0], height=resolution[1], hue_num=hue_num)
            fname = f"./bt2020_P3_hue_chroma_{width}x{height}_"
            fname += f"h_num-{hue_num}.png"
            print(fname)
            tpg.img_wirte_float_as_16bit_int(fname, img)
            fname_with_profile = str('img' / Path(fname))
            tpg.img_wirte_float_as_16bit_int(fname, img)
            cmd = [
                'convert', fname, '-profile', icc_profile, fname_with_profile]
            subprocess.run(cmd)
            os.remove(fname)

    # h_val = 104
    # h_idx = int(h_val)
    # debug_hue_chroma_pattern_core(
    #     inner_lut=inner_lut, outer_lut=outer_lut, h_idx=h_idx, h_val=h_val,
    #     chroma_num=14)
    # plot_hue_l_focal_plane(inner_lut=inner_lut, outer_lut=outer_lut)

    # plot_cl_plane_using_intp_for_blog_make_lut_image(
    #     h_idx=40, h_val=40, color_space_name=cs.BT2020)
    # plot_cl_plane_using_intp_for_blog_make_lut_image(
    #     h_idx=98, h_val=99, color_space_name=cs.BT2020)
    # plot_cl_plane_using_intp_for_blog_make_lut_image(
    #     h_idx=99, h_val=99, color_space_name=cs.BT2020)
    # plot_cl_plane_using_intp_for_blog_make_lut_image(
    #     h_idx=100, h_val=100, color_space_name=cs.BT2020)
