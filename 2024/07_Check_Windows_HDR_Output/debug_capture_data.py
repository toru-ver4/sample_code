# -*- coding: utf-8 -*-
"""

==========

"""

# import standard libraries
import os
from pathlib import Path
import subprocess

# import third-party libraries
import numpy as np
import color_space as cs
from colour import XYZ_to_xyY

# import my libraries
import test_pattern_generator2 as tpg
import transfer_functions as tf
import plot_utility as pu
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def get_wrgbmyc_data(fname="./debug/capture/dummy.png"):
    img = np.round(tpg.img_read_as_float(fname) * 1023).astype(np.uint16)
    st_pos_h = 120
    st_pos_v = 1678
    ed_pos_h = 3716
    ed_pos_v = 2000
    num_of_h_sample = 65
    num_of_v_sample = 7
    pos_h_list = np.linspace(st_pos_h, ed_pos_h, num_of_h_sample)
    pos_v_list = np.linspace(st_pos_v, ed_pos_v, num_of_v_sample)

    pos_list = np.zeros((num_of_v_sample, num_of_h_sample, 2), dtype=np.uint16)

    for idx, pos_v in enumerate(pos_v_list):
        h_list = pos_h_list
        v_list = np.ones_like(h_list) * pos_v
        temp_pos_list = np.column_stack((h_list, v_list))
        pos_list[idx] = temp_pos_list

    wrgbmyc = img[pos_list[..., 1], pos_list[..., 0]]

    return wrgbmyc


def plot_tone_mapping_characteristics(fname):
    max_10bit = 1023
    wrgbymc = get_wrgbmyc_data(fname=fname)
    w_green = wrgbymc[0, :, 1]
    w_green_lumi = tf.eotf_to_luminance(w_green / max_10bit, tf.ST2084)

    x_cv = np.array([x * 16 for x in range(64)] + [1023])
    x_ref_lumi = tf.eotf_to_luminance(x_cv/max_10bit, tf.ST2084)
    y_ref_lumi = x_ref_lumi.copy()

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=None,
        graph_title_size=None,
        xlabel="Target luminance (nits)",
        ylabel="Captured luminance (nits)",
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
    pu.log_scale_settings(ax1=ax1, bg_color="#F0F0F0", grid_color="#808080")
    ax1.plot(x_ref_lumi, y_ref_lumi, '--', color='k', label="Reference", lw=2)
    ax1.plot(
        x_ref_lumi, w_green_lumi, color=pu.RED,
        label="Captured luminance"
    )
    pu.show_and_save(
        fig=fig, legend_loc='upper left', show=False,
        save_fname='./img/edge_tone_mapping_characteristics')


def plot_gamut_mapping_characteristics():
    rate = 1.0
    xmin = -0.1
    xmax = 0.8
    ymin = -0.1
    ymax = 1.0
    st_wl = 380
    ed_wl = 780
    wl_step = 1
    plot_wl_list = [
        410, 450, 470, 480, 485, 490, 495,
        500, 505, 510, 520, 530, 540, 550, 560, 570, 580, 590,
        600, 620, 690]
    cmf_xy = pu.calc_horseshoe_chromaticity(
        st_wl=st_wl, ed_wl=ed_wl, wl_step=wl_step)
    cmf_xy_norm = pu.calc_normal_pos(
        xy=cmf_xy, normal_len=0.05, angle_degree=90)
    wl_list = np.arange(st_wl, ed_wl + 1, wl_step)
    xy_image = pu.get_chromaticity_image(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, cmf_xy=cmf_xy)

    max_10bit = 1023
    wrgbymc = get_wrgbmyc_data(fname=fname)
    rgb = wrgbymc[1:4, -1]
    large_xyz = cs.rgb_to_large_xyz(
        rgb=tf.eotf(rgb/max_10bit, tf.ST2084),
        color_space_name=cs.BT2020
    )
    xyY = XYZ_to_xyY(large_xyz)
    after_primaries = pu.add_first_value_to_end(xyY[..., :2])
    print(after_primaries)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20 * rate,
        figsize=((xmax - xmin) * 10 * rate,
                 (ymax - ymin) * 10 * rate),
        graph_title="Chromaticity Gamut",
        graph_title_size=None,
        xlabel=None, ylabel=None,
        axis_label_size=None,
        legend_size=14 * rate,
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
        xtick=[x * 0.1 + xmin for x in
               range(int((xmax - xmin)/0.1) + 1)],
        ytick=[x * 0.1 + ymin for x in
               range(int((ymax - ymin)/0.1) + 1)],
        xtick_size=17 * rate,
        ytick_size=17 * rate,
        linewidth=4 * rate,
        minor_xtick_num=2,
        minor_ytick_num=2)
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=2*rate, label=None)
    for idx, wl in enumerate(wl_list):
        if wl not in plot_wl_list:
            continue
        pu.draw_wl_annotation(
            ax1=ax1, wl=wl, rate=rate,
            st_pos=[cmf_xy_norm[idx, 0], cmf_xy_norm[idx, 1]],
            ed_pos=[cmf_xy[idx, 0], cmf_xy[idx, 1]])
    bt2020_gamut = pu.get_primaries(name=cs.BT2020)
    ax1.plot(bt2020_gamut[:, 0], bt2020_gamut[:, 1], '-o',
             c=pu.RED, label="BT.2020", lw=5*rate)

    ax1.plot(after_primaries[:, 0], after_primaries[:, 1], '-o',
             c=pu.BLUE, label="Captured Chromaticity Gamut", lw=1.2*rate)

    ap0_gamut = pu.get_primaries(name=cs.ACES_AP0)
    ax1.plot(ap0_gamut[:, 0], ap0_gamut[:, 1], '--k',
             label="ACES AP0", lw=1*rate)
    ax1.plot(
        [0.3127], [0.3290], 'x', label='D65', ms=12*rate, mew=2*rate,
        color='k', alpha=0.8)
    ax1.imshow(xy_image, extent=(xmin, xmax, ymin, ymax), alpha=0.5)
    pu.show_and_save(
        fig=fig, legend_loc='upper right', show=True,
        save_fname="./img/edge_gamut_mapping.png")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    fname = "./debug/capture/edge_src_10000_momitor_1-400_400_BT709.png"
    plot_tone_mapping_characteristics(fname=fname)
    plot_gamut_mapping_characteristics()
