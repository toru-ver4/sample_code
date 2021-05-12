# -*- coding: utf-8 -*-
"""
create gamut boundary lut.
"""

# import standard libraries
import os
import time
from colour.models.cie_lab import XYZ_to_Lab
from colour.models.rgb.rgb_colourspace import RGB_to_XYZ

# import third-party libraries
import numpy as np
from colour import LCHab_to_Lab, Lab_to_XYZ, XYZ_to_RGB, RGB_COLOURSPACES

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


def debug_mplot_cl_plane_from_image():
    ll_line = 97.4
    x = [0, 200]
    y = [ll_line, ll_line]
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(20, 16),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Chroma-Lightness Plane",
        graph_title_size=None,
        xlabel="Chroma", ylabel="Lightness",
        axis_label_size=None,
        legend_size=17,
        xlim=[0, 220],
        ylim=[0, 100],
        xtick=None,
        ytick=[x * 20 for x in range(5)] + [ll_line],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)

    rgb_img = tpg.img_read_as_float(
        "./img/ITU-R BT.2020_h-99.0_c-2560_l_1440.png")
    ax1.imshow(
        rgb_img, extent=(0, 220, 0, 100),
        aspect='auto')
    ax1.plot(x, y, '--k', lw=2, alpha=0.5)
    pu.show_and_save(
        fig=fig, legend_loc=None, save_fname="./img/test_cl.png")


def create_valid_ab_plane_image_srgb(
        l_val=50, ab_max=200, ab_sample=512, color_space_name=cs.BT2020):
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
    dummy_rgb = np.array([0.5, 0.5, 0.5])
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
        fig=fig, legend_loc=None, plot=False,
        save_fname=f"./img/test_ab_{idx:04d}.png")


def plot_ab_plane_core(ty_ch_lut, l_val):
    """
    Parameters
    ----------
    ty_ch_lut : ndarray
        gamut boundary data.
        shape is (Hue_num, 3).
        the data order is L*, C*, Hab
    l_val : float
        Lightness Value of the ab plane
    """
    pass


def plot_ab_plane_seq(ty_lch_lut):
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
    block_process_num = 20
    block_num = int(total_process_num / block_process_num + 0.5)

    for b_idx in range(block_num):
        for p_idx in range(block_process_num):
            l_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={l_idx}")  # User
            if l_idx >= total_process_num:                         # User
                break
            args = dict(                                           # User
                ty_ch_lut=ty_lch_lut[l_idx], l_val=l_idx/(l_num-1))
            plot_ab_plane_core(**args)


def check_lch_2d_lut():
    lut = np.load("./lut_sample_50_361_8192.npy")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # lut = np.load("./lut_sample_50_361_8192.npy")
    # plot_ab_plane_seq(ty_lch_lut=lut)
    debug_plot_cl_plane_core(hue=99)
    # for idx in range(200):
    #     l_val = 90 + 0.05 * idx
    #     print(f"idx={idx}, l_val={l_val}")
    #     debug_plot_ab_plane(
    #         l_val=l_val, color_space_name=cs.BT2020, idx=idx)
    # srgb_img = create_valid_ab_plane_image_srgb(
    #     l_val=90, ab_max=200, ab_sample=512, color_space_name=cs.BT2020)
    # tpg.img_wirte_float_as_16bit_int("uuu.png", srgb_img)