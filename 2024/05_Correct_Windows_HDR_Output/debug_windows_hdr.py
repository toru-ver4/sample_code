# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from pathlib import Path

# import third-party libraries
import numpy as np
from imagecodecs import JPEGXR, imread
from colour.io import write_image
import matplotlib.pyplot as plt

# import my libraries
import plot_utility as pu
import color_space as cs
import transfer_functions as tf
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def get_wgbmyc_corrdinate():
    num_of_color = 7
    num_of_cv = 65
    h_st = 122
    h_ed = 3716
    v_st = 1680
    v_ed = 2004
    v_list = np.round(np.linspace(v_st, v_ed, num_of_color)).astype(np.uint16)
    h_list = np.round(np.linspace(h_st, h_ed, num_of_cv)).astype(np.uint16)
    xy_list = []
    for v_pos in v_list:
        line = [[h_pos, v_pos] for h_pos in h_list]
        xy_list.append(line)

    return xy_list


def get_pixel_values(img, xy_pos_list):
    values = []
    for xy_pos in xy_pos_list:
        # print(f"xy_pos = {xy_pos}")
        value = img[xy_pos[1], xy_pos[0]]
        values.append(value)
    values = np.array(values)
    
    return values


def plot_captured_hdr_tp():
    xy_list = get_wgbmyc_corrdinate()
    jsr_file = "./Windows_HDR_Capture/YouTube_10000_gain_1.0.jxr"
    basename = Path(jsr_file).stem
    sc_rgb_img = imread(jsr_file)[..., :3]
    rec2020_rgb_img = conv_scRGB_to_rec2020(sc_rgb_img=sc_rgb_img)
    xx = np.linspace(0, 1024, 65).astype(np.uint16)
    xx[-1] = xx[-1] - 1

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Title",
        graph_title_size=None,
        xlabel="X Axis Label",
        ylabel="Y Axis Label",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=[x * 128 for x in range(8)] + [1023],
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    rgb = get_pixel_values(img=rec2020_rgb_img, xy_pos_list=xy_list[0])
    rr = rgb[1:, 0]
    gg = rgb[1:, 1]
    bb = rgb[1:, 2]

    rr_ratio = rr / rr
    gg_ratio = gg / rr
    bb_ratio = bb / rr

    ax1.plot(xx[1:], rr, '-o', color=pu.RED, label="R")
    ax1.plot(xx[1:], gg, '-o', color=pu.GREEN, label="G")
    ax1.plot(xx[1:], bb, '-o', color=pu.BLUE, label="B")

    # ax1.plot(xx[1:], rr_ratio, '-o', color=pu.RED, label="R")
    # ax1.plot(xx[1:], gg_ratio, '-o', color=pu.GREEN, label="G")
    # ax1.plot(xx[1:], bb_ratio, '-o', color=pu.BLUE, label="B")
    # pu.log_sacle_settings_x_linear_y_log(ax=ax1)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', show=False,
        save_fname=f"./img/{basename}.png")
    

def conv_scRGB_to_rec2020(sc_rgb_img):
    large_xyz = cs.rgb_to_large_xyz(rgb=sc_rgb_img, color_space_name=cs.sRGB)
    rec2020_rgb_img = cs.large_xyz_to_rgb(
        xyz=large_xyz, color_space_name=cs.BT2020)
    return rec2020_rgb_img


def conv_hdr_tp_from_sc_rgb_to_rec2100_pq(
        jsr_file="./Windows_HDR_Capture/bak/YouTube_10000_gain_1.0.jxr"):
    xy_list = get_wgbmyc_corrdinate()
    sc_rgb_img = imread(jsr_file)[..., :3]  # remove alpha channel
    rec2020_rgb_img = conv_scRGB_to_rec2020(sc_rgb_img=sc_rgb_img)

    # extract ramp tp pixels
    rgb = []
    for idx in range(7):
        rgb_temp\
            = get_pixel_values(img=rec2020_rgb_img, xy_pos_list=xy_list[idx])
        rgb.append(rgb_temp)
    rgb = np.array(rgb)

    # linear to ST2084
    rgb = np.clip(rgb, 0, 1000000)
    rgb_st2084 = tf.oetf_from_luminance(rgb * 100, tf.ST2084)
    rgb_st2084 = np.round(rgb_st2084 * 1023).astype(np.uint16)

    return rgb_st2084


def plot_tp_7colors(jxr_file):
    basename = Path(jxr_file).stem
    title_base = basename.replace("TP_Rec2020_", "")
    xy_list = get_wgbmyc_corrdinate()
    img_st2084 = conv_hdr_tp_from_sc_rgb_to_rec2100_pq(jsr_file=jxr_file)
    fig, axes = plt.subplots(7, 1, figsize=(6, 16))  # 7行1列、図のサイズを指定
    xx = np.linspace(0, 1024, 65).astype(np.uint16)
    title_list = [
        "White", "Red", "Green", "Blue", "Majenta", "Yellow", "Cyan"
    ]

    for idx in range(7):
        ax1 = axes[idx]
        yy = img_st2084[idx]
        ms = 4
        ax1.plot(xx, yy[..., 0], '-o', ms=ms, color=pu.RED, label="R")
        ax1.plot(xx, yy[..., 1], '-o', ms=ms, color=pu.GREEN, label="G")
        ax1.plot(xx, yy[..., 2], '-o', ms=ms, color=pu.BLUE, label="B")
        ax1.set_xticks([x * 128 for x in range(8)] + [1023])
        ax1.set_yticks([x * 256 for x in range(4)] + [1023])
        ax1.set_xlim([-10, 1033])
        ax1.set_ylim([-20, 1043])
        ax1.set_title(f'{title_base} - {title_list[idx]} Patch')
        ax1.grid(True)
        ax1.legend(loc='upper left')

    plt.tight_layout()

    save_fname = f"./img/{basename}.png"
    print(save_fname)
    plt.savefig(save_fname, dpi=100)

    # plt.show()


def plot_concat_debug_player_result():
    jxr_file_list = [
        "./Windows_HDR_Capture/gain_1.0/TP_Rec2020_Edge.jxr",
        "./Windows_HDR_Capture/gain_1.0/TP_Rec2020_Chrome.jxr",
        "./Windows_HDR_Capture/gain_1.0/TP_Rec2020_Movies-TV.jxr",
        "./Windows_HDR_Capture/gain_1.0/TP_Rec2020_MPC-BE.jxr",
        "./Windows_HDR_Capture/gain_1.0/TP_Rec2020_VLC.jxr"
    ]
    for jxr_file in jxr_file_list:
        plot_tp_7colors(jxr_file=jxr_file)

    img_list = []
    for jxr_file in jxr_file_list:
        basename = Path(jxr_file).stem
        in_fname = f"./img/{basename}.png"
        img = tpg.img_read_as_float(in_fname)
        img_list.append(img)
    out_img = np.hstack(img_list)
    concat_fname = "./img/concat_player_result.png"
    print(concat_fname)
    tpg.img_wirte_float_as_16bit_int(concat_fname, out_img)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # plot_captured_hdr_tp()
    # conv_hdr_tp_from_sc_rgb_to_rec2100_pq()
    # plot_tp_7colors(jsr_file="./Windows_HDR_Capture/bak/600.jxr")
    plot_concat_debug_player_result()
