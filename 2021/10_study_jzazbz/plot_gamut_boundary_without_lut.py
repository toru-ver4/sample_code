# -*- coding: utf-8 -*-
"""
plot gamut boundary
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import XYZ_to_RGB, RGB_COLOURSPACES
from multiprocessing import Pool, cpu_count

# import my libraries
import color_space as cs
import transfer_functions as tf
from jzazbz import jzazbz_to_large_xyz, st2084_oetf_like, st2084_eotf_like
from create_gamut_booundary_lut import is_out_of_gamut_rgb
from test_pattern_generator2 import img_wirte_float_as_16bit_int
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def create_valid_ab_plane_image_st2084(
        j_val=0.5, ab_max=1.0, ab_sample=512, color_space_name=cs.BT2020,
        bg_rgb_luminance=np.array([50, 50, 50])):
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

    ng_idx = is_out_of_gamut_rgb(rgb=rgb_luminance/10000)
    rgb_luminance[ng_idx] = bg_rgb_luminance
    rgb_st2084 = tf.oetf_from_luminance(
        np.clip(rgb_luminance, 0.0, 10000), tf.ST2084)

    return rgb_st2084


def plot_ab_plane_st2084(
        j_idx=0, j_val=0.5, ab_max=1.0, ab_sample=1536,
        color_space_name=cs.BT2020):
    rgb_st2084 = create_valid_ab_plane_image_st2084(
        j_val=j_val, ab_max=ab_max, ab_sample=ab_sample,
        color_space_name=color_space_name)
    luminance = int(
        round(st2084_eotf_like(j_val)) + 0.5)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(16, 16),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=f"Jz={j_val:.2f},  Luminance={luminance} [cd/m2]?",
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
    ax1.plot([0], [0], '.')
    ax1.imshow(
        rgb_st2084, extent=(-ab_max, ab_max, -ab_max, ab_max), aspect='auto')
    fname = "/work/overuse/2021/10_jzazbz/img_seq_ab/"
    fname += f"azbz_plane_{color_space_name}_{j_idx}.png"
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)


def thread_wrapper_plot_ab_plane_st2084(args):
    plot_ab_plane_st2084(**args)


def plot_ab_plane_seq(color_space_name):
    """
    Parameters
    ----------
    ty_lch_lut : ndarray
        gamut boundary data.
        shape is (Lightness_num, Hue_num, 3).
        the data order is L*, C*, Hab
    """
    j_num = 501

    total_process_num = j_num
    block_process_num = cpu_count() // 4
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            j_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={j_idx}")  # User
            if j_idx >= total_process_num:                         # User
                break
            d = dict(
                j_idx=j_idx, j_val=j_idx/(j_num-1), ab_max=0.5, ab_sample=1536,
                color_space_name=color_space_name)
            # plot_ab_plane_st2084(**d)
            args.append(d)
        #     break
        # break
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_plot_ab_plane_st2084, args)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    luminance = 100
    j_val = st2084_oetf_like(luminance)
    # img = create_valid_ab_plane_image_st2084(
    #     j_val=j_val, ab_max=0.5, ab_sample=1024, color_space_name=cs.BT2020,
    #     bg_rgb_luminance=np.array([20, 20, 20]))
    # img_wirte_float_as_16bit_int(f"./test_lum-{luminance}.png", img)

    # plot_ab_plane_st2084(
    #     j_idx=0, j_val=j_val, ab_max=0.5, ab_sample=1024,
    #     color_space_name=cs.BT2020)

    plot_ab_plane_seq(color_space_name=cs.BT2020)
