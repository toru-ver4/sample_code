# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from pathlib import Path

# import third-party libraries
import numpy as np
from imagecodecs import JPEGXR, imread
from colour import matrix_RGB_to_RGB, normalised_primary_matrix
from colour.models import RGB_COLOURSPACE_BT709, RGB_COLOURSPACE_BT2020
from colour.io import write_image
from colour.utilities import tstack
import matplotlib.pyplot as plt
from scipy import linalg

# import my libraries
import plot_utility as pu
import color_space as cs
import transfer_functions as tf
import test_pattern_generator2 as tpg
from create_10bit_ramp_tp import calc_ramp_pattern_block_st_pos_with_color_idx, \
    NUM_OF_CODE_VALUE

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
    jxr_file = "./Windows_HDR_Capture/YouTube_10000_gain_1.0.jxr"
    basename = Path(jxr_file).stem
    sc_rgb_img = imread(jxr_file)[..., :3]
    rec2020_rgb_img = conv_scRGB_to_target_rgb(sc_rgb_img=sc_rgb_img)
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
    

def conv_scRGB_to_target_rgb(sc_rgb_img, target_color_space_name=cs.BT2020):
    large_xyz = cs.rgb_to_large_xyz(rgb=sc_rgb_img, color_space_name=cs.sRGB)
    target_rgb = cs.large_xyz_to_rgb(
        xyz=large_xyz, color_space_name=target_color_space_name)
    return target_rgb


def conv_hdr_tp_from_sc_rgb_to_rec2100_pq(
        jsr_file="./Windows_HDR_Capture/bak/YouTube_10000_gain_1.0.jxr"):
    xy_list = get_wgbmyc_corrdinate()
    sc_rgb_img = imread(jsr_file)[..., :3]  # remove alpha channel
    rec2020_rgb_img = conv_scRGB_to_target_rgb(sc_rgb_img=sc_rgb_img)

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


def conv_hdr_tp_from_sc_rgb_to_target_sc_pq_10bit_ramp(
        jsr_file="./Windows_HDR_Capture/bak/YouTube_10000_gain_1.0.jxr",
        target_color_space_name=cs.BT2020):
    sc_rgb_img = imread(jsr_file)[..., :3].astype(np.float64)  # remove alpha channel
    target_rgb_img = conv_scRGB_to_target_rgb(
        sc_rgb_img=sc_rgb_img, target_color_space_name=target_color_space_name)
    
    #################################
    # remove !!!
    target_rgb_img = sc_rgb_img.copy()
    #################################

    block_size = 32
    width = 3840

    # extract ramp tp pixels
    rgb = []
    for idx in range(7):
        for cv in range(1024):
            st_pos\
                = calc_ramp_pattern_block_st_pos_with_color_idx(
                    code_value=cv, width=width, block_size=block_size,
                    color_kind_idx=idx)
            capture_pos = [
                st_pos[0] + (block_size // 2), st_pos[1] + (block_size // 2)]
            rgb.append(target_rgb_img[capture_pos[1], capture_pos[0]])
    rgb = np.array(rgb).reshape(-1, 1024, 3)
    print(rgb.shape)

    # linear to ST2084
    rgb = np.clip(rgb, 0, 1000000)
    for idx, rr in enumerate(rgb[2, ..., 0]):
        print(idx, rr)
    rgb_st2084 = tf.oetf_from_luminance(rgb * 100, tf.ST2084)
    rgb_st2084 = np.round(rgb_st2084 * 1023).astype(np.uint16)

    return rgb_st2084


def plot_tp_7colors(jxr_file):
    basename = Path(jxr_file).stem
    title_base = basename.replace("TP_Rec2020_", "")
    xy_list = get_wgbmyc_corrdinate()
    img_st2084 = conv_hdr_tp_from_sc_rgb_to_rec2100_pq(jsr_file=jxr_file)
    fig, axes = plt.subplots(7, 1, figsize=(6, 16))
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


def check_half_float_error():
    x = np.arange(0, 1025, 16)
    x[-1] = x[-1] - 1
    x = tstack([x, x, x])
    color_mask_list = np.array([
        [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 0, 1], [1, 1, 0], [0, 1, 1]
    ])
    yy = []
    for color_mask in color_mask_list:
        yy_temp = x * color_mask
        yy.append(yy_temp)
    yy = np.array(yy) / 1023

    yy_luminance = tf.eotf_to_luminance(yy, tf.ST2084)
    yy_float16 = yy_luminance.astype(np.uint16)
    yy_float64 = yy_luminance.copy()
    diff = np.abs(yy_float64 - yy_float16)
    diff_st2084 = tf.oetf_from_luminance(diff, tf.ST2084)
    diff_st2084 = np.round(diff_st2084 * 1023).astype(np.uint16)

    fig, axes = plt.subplots(7, 1, figsize=(6, 16))
    xx = np.linspace(0, 1024, 65).astype(np.uint16)
    title_list = [
        "White", "Red", "Green", "Blue", "Majenta", "Yellow", "Cyan"
    ]

    for idx in range(7):
        ax1 = axes[idx]
        yy = diff_st2084[idx]
        ms = 4
        ax1.plot(xx, yy[..., 0], '-o', ms=ms, color=pu.RED, label="R")
        ax1.plot(xx, yy[..., 1], '-o', ms=ms, color=pu.GREEN, label="G")
        ax1.plot(xx, yy[..., 2], '-o', ms=ms, color=pu.BLUE, label="B")
        ax1.set_xticks([x * 128 for x in range(8)] + [1023])
        ax1.set_yticks([x * 256 for x in range(4)] + [1023])
        ax1.set_xlim([-10, 1033])
        ax1.set_ylim([-20, 1043])
        ax1.set_title(f'{title_list[idx]} np.float64 - np.float16')
        ax1.grid(True)
        ax1.legend(loc='upper left')

    plt.tight_layout()
    save_fname = f"./img/half_float_diff.png"
    print(save_fname)
    plt.savefig(save_fname, dpi=100)


def plot_10bit_wrgbmyc_ramp_data(
        jxr_file, replace_str="TP_Rec2020_10-bit_",
        target_color_space_name=cs.BT2020):
    basename = Path(jxr_file).stem
    title_base = basename.replace(replace_str, "")
    img_st2084 = conv_hdr_tp_from_sc_rgb_to_target_sc_pq_10bit_ramp(
        jsr_file=jxr_file, target_color_space_name=target_color_space_name)
    fig, axes = plt.subplots(7, 1, figsize=(8, 20))
    xx = np.arange(1024).astype(np.uint16)
    title_list = [
        "White", "Red", "Green", "Blue", "Majenta", "Yellow", "Cyan"
    ]

    for idx in range(7):
        ax1 = axes[idx]
        yy = img_st2084[idx]
        ms = 4
        ax1.plot(xx, yy[..., 0], '-', ms=ms, color=pu.RED, label="R")
        ax1.plot(xx, yy[..., 1], '-', ms=ms, color=pu.GREEN, label="G")
        ax1.plot(xx, yy[..., 2], '-', ms=ms, color=pu.BLUE, label="B")
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


def plot_rec2020_10bit_wrgbmyc_ramp_data_all():
    jxr_file_list = [
        "./Windows_HDR_Capture/gain_1.0_10-bit/TP_Rec2020_10-bit_Chrome.jxr",
        "./Windows_HDR_Capture/gain_1.0_10-bit/TP_Rec2020_10-bit_Edge.jxr",
        "./Windows_HDR_Capture/gain_1.0_10-bit/TP_Rec2020_10-bit_MPC-BE.jxr",
        "./Windows_HDR_Capture/gain_1.0_10-bit/TP_Rec2020_10-bit_VLC.jxr",
    ]

    for jxr_file in jxr_file_list:
        plot_10bit_wrgbmyc_ramp_data(
            jxr_file=jxr_file, replace_str="TP_Rec2020_10-bit_")

    img_list = []
    for jxr_file in jxr_file_list:
        basename = Path(jxr_file).stem
        in_fname = f"./img/{basename}.png"
        img = tpg.img_read_as_float(in_fname)
        img_list.append(img)
    out_img = np.hstack(img_list)
    concat_fname = "./img/concat_rec2020_10bit_wrgbmyc_result.png"
    print(concat_fname)
    tpg.img_wirte_float_as_16bit_int(concat_fname, out_img)


def plot_rec709_10bit_wrgbmyc_ramp_data_all():
    jxr_file_list = [
        # "./Windows_HDR_Capture/gain_1.0_10-bit/TP_Rec709_10-bit_Chrome.jxr",
        "./Windows_HDR_Capture/gain_1.0_10-bit/TP_Rec709_10-bit_Edge.jxr",
        # "./Windows_HDR_Capture/gain_1.0_10-bit/TP_Rec709_10-bit_MPC-BE.jxr",
        # "./Windows_HDR_Capture/gain_1.0_10-bit/TP_Rec709_10-bit_VLC.jxr",
    ]

    for jxr_file in jxr_file_list:
        plot_10bit_wrgbmyc_ramp_data(
            jxr_file=jxr_file, replace_str="TP_Rec709_10-bit_",
            target_color_space_name=cs.BT709)

    img_list = []
    for jxr_file in jxr_file_list:
        basename = Path(jxr_file).stem
        in_fname = f"./img/{basename}.png"
        img = tpg.img_read_as_float(in_fname)
        img_list.append(img)
    out_img = np.hstack(img_list)
    concat_fname = "./img/concat_rec709_10bit_wrgbmyc_result.png"
    print(concat_fname)
    tpg.img_wirte_float_as_16bit_int(concat_fname, out_img)


def debug_plot_check_raw():
    width = 3840
    block_size = 32
    # jxr_flle = "./Windows_HDR_Capture/gain_1.0_10-bit/TP_Rec709_10-bit_Edge.jxr"
    # jxr_flle = "./Windows_HDR_Capture/gain_1.0_10-bit/TP_Rec709_10-bit_Movies_and_TV.jxr"
    jxr_flle = "./Windows_HDR_Capture/gain_1.0_10-bit/TP_Rec2020_10-bit_Edge.jxr"
    sc_rgb_img = imread(jxr_flle)[..., :3].astype(np.float64)

    basename = Path(jxr_flle).stem

    ######################
    # please change
    # target_rgb_img = conv_scRGB_to_target_rgb(
    #     sc_rgb_img=sc_rgb_img, target_color_space_name=cs.BT2020)
    target_rgb_img = sc_rgb_img.copy()
    ######################

    rgb = []
    for idx in range(7):
        for cv in range(1024):
            st_pos\
                = calc_ramp_pattern_block_st_pos_with_color_idx(
                    code_value=cv, width=width, block_size=block_size,
                    color_kind_idx=idx)
            capture_pos = [
                st_pos[0] + (block_size // 2), st_pos[1] + (block_size // 2)]
            rgb.append(target_rgb_img[capture_pos[1], capture_pos[0]])
    rgb = np.array(rgb).reshape(-1, 1024, 3)

    print(np.min(rgb), np.max(rgb))

    ######################
    # please change
    # rgb[rgb < 0] = 0
    # yy_for_plot = tf.oetf_from_luminance(rgb * 100, tf.ST2084)
    yy_for_plot = rgb
    title_base = f"{basename}"
    ######################

    # # multi plots
    # fig, axes = plt.subplots(7, 1, figsize=(10, 19))
    # xx = np.arange(NUM_OF_CODE_VALUE)
    # title_list = [
    #     "White", "Red", "Green", "Blue", "Majenta", "Yellow", "Cyan"
    # ]
    # for idx in range(7):
    #     ax1 = axes[idx]
    #     yy = yy_for_plot[idx]
    #     ms = 4
    #     ax1.plot(xx, yy[..., 0], '-', ms=ms, color=pu.RED, label="R")
    #     ax1.plot(xx, yy[..., 1], '-', ms=ms, color=pu.GREEN, label="G")
    #     ax1.plot(xx, yy[..., 2], '-', ms=ms, color=pu.BLUE, label="B")
    #     ax1.set_xticks([x * 128 for x in range(8)] + [1023])
    #     # ax1.set_yticks([x * 256 for x in range(4)] + [1023])
    #     # ax1.set_xlim([-10, 1033])
    #     # ax1.set_ylim([-0.8, 0.8])
    #     ax1.set_title(f'{title_base} - {title_list[idx]} Patch')
    #     ax1.grid(True)
    #     ax1.legend(loc='upper left')
    # plt.tight_layout()
    # save_fname = f"./debug_img/debug_{basename}.png"
    # print(save_fname)
    # plt.savefig(save_fname, dpi=100)

    # # single plots
    # xx = np.arange(NUM_OF_CODE_VALUE)
    # fig, ax1 = pu.plot_1_graph(
    #     fontsize=20,
    #     figsize=(20, 15),
    #     bg_color=(0.96, 0.96, 0.96),
    #     graph_title="Title",
    #     graph_title_size=None,
    #     xlabel="X Axis Label",
    #     ylabel="Y Axis Label",
    #     axis_label_size=None,
    #     legend_size=17,
    #     xlim=[-10, 1033],
    #     ylim=None,
    #     xtick=([x * 128 for x in range(8)] + [1023]),
    #     ytick=None,
    #     xtick_size=None, ytick_size=None,
    #     linewidth=3,
    #     minor_xtick_num=None,
    #     minor_ytick_num=None)
    # ax1.plot(xx, yy_for_plot[5, :, 0], color=pu.RED, label="R")
    # # pu.log_sacle_settings_x_linear_y_log(ax=ax1)
    # pu.show_and_save(
    #     fig=fig, legend_loc='upper left',
    #     save_fname=f"./debug_img/debug_{basename}_Yellow.png"
    # )

    # single plots
    xx = np.arange(NUM_OF_CODE_VALUE)
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(32, 15),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Title",
        graph_title_size=None,
        xlabel="Code Value (10-bit)",
        ylabel="(Yn+1 - Yn) / average(Yn+1 + Yn)",
        axis_label_size=None,
        legend_size=17,
        xlim=[-10, 1033],
        ylim=[-0.05, 0.2],
        xtick=([x * 128 for x in range(8)] + [1023]),
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    yy = yy_for_plot[1, :, 0]
    diff = yy[1:] - yy[:-1]
    base = (yy[1:] + yy[:-1]) / 2.0
    base = tf.eotf_to_luminance(xx[1:]/1023, tf.ST2084) / 10000 * 124
    rate = diff / base
    ax1.plot(xx[1:], rate, color=pu.RED, label="R")
    # pu.log_sacle_settings_x_linear_y_log(ax=ax1)
    pu.show_and_save(
        fig=fig, legend_loc='upper left',
        save_fname=f"./debug_img/debug_{basename}_Red_diff.png"
    )


def debug_plot_check_after_conv():
    width = 3840
    block_size = 32
    # jxr_flle = "./Windows_HDR_Capture/gain_1.0_10-bit/TP_Rec709_10-bit_Edge.jxr"
    # jxr_flle = "./Windows_HDR_Capture/gain_1.0_10-bit/TP_Rec709_10-bit_avif_Edge.jxr"
    # jxr_flle = "./Windows_HDR_Capture/gain_1.0_10-bit/TP_Rec709_10-bit_Movies_and_TV.jxr"
    # jxr_flle = "./Windows_HDR_Capture/gain_1.0_10-bit/TP_Rec2020_10-bit_Edge.jxr"
    jxr_flle = "./Windows_HDR_Capture/gain_1.0_10-bit/TP_Rec2020_10-bit_avif_Edge.jxr"
    sc_rgb_img = imread(jxr_flle)[..., :3].astype(np.float64)

    basename = Path(jxr_flle).stem

    ######################
    # please change
    target_rgb_img = conv_scRGB_to_target_rgb(
        sc_rgb_img=sc_rgb_img, target_color_space_name=cs.BT2020)
    # target_rgb_img = sc_rgb_img.copy()
    ######################

    rgb = []
    for idx in range(7):
        for cv in range(1024):
            st_pos\
                = calc_ramp_pattern_block_st_pos_with_color_idx(
                    code_value=cv, width=width, block_size=block_size,
                    color_kind_idx=idx)
            capture_pos = [
                st_pos[0] + (block_size // 2), st_pos[1] + (block_size // 2)]
            rgb.append(target_rgb_img[capture_pos[1], capture_pos[0]])
    rgb = np.array(rgb).reshape(-1, 1024, 3)

    print(np.min(rgb), np.max(rgb))

    ######################
    # please change
    rgb[rgb < 0] = 0
    yy_for_plot = tf.oetf_from_luminance(rgb * 100, tf.ST2084)
    # yy_for_plot = rgb
    title_base = f"{basename}"
    ######################

    # multi plots
    fig, axes = plt.subplots(7, 1, figsize=(10, 19))
    xx = np.arange(NUM_OF_CODE_VALUE)
    title_list = [
        "White", "Red", "Green", "Blue", "Majenta", "Yellow", "Cyan"
    ]
    for idx in range(7):
        ax1 = axes[idx]
        yy = yy_for_plot[idx]
        ms = 4
        ax1.plot(xx, yy[..., 0], '-', ms=ms, color=pu.RED, label="R")
        ax1.plot(xx, yy[..., 1], '-', ms=ms, color=pu.GREEN, label="G")
        ax1.plot(xx, yy[..., 2], '-', ms=ms, color=pu.BLUE, label="B")
        ax1.set_xticks([x * 128 for x in range(8)] + [1023])
        # ax1.set_yticks([x * 256 for x in range(4)] + [1023])
        # ax1.set_xlim([-10, 1033])
        # ax1.set_ylim([-0.8, 0.8])
        ax1.set_title(f'{title_base} - {title_list[idx]} Patch')
        ax1.grid(True)
        ax1.legend(loc='upper left')
    plt.tight_layout()
    save_fname = f"./debug_img/debug_{basename}_after_conv.png"
    print(save_fname)
    plt.savefig(save_fname, dpi=100)


def debug_check_srgb_rgbw():
    red_file = "./Windows_HDR_Capture/sRGB_rgbw/sRGB_red.jxr"
    green_file = "./Windows_HDR_Capture/sRGB_rgbw/sRGB_green.jxr"
    blue_file = "./Windows_HDR_Capture/sRGB_rgbw/sRGB_blue.jxr"
    white_file = "./Windows_HDR_Capture/sRGB_rgbw/sRGB_white.jxr"
    file_list = [red_file, green_file, blue_file, white_file]
    color_name_list = ["R", "G", "B", "W"]

    for file_name, color_name in zip(file_list, color_name_list):
        sc_rgb_img = imread(file_name)[..., :3].astype(np.float64)
        height, width = sc_rgb_img.shape[:2]
        pos_v = height // 2
        pos_h = width // 2
        rgb_val = sc_rgb_img[pos_v, pos_h]
        print(f"{color_name} - {rgb_val}")


def debug_output_matrix():
    rec2020_to_rec709_mtx = matrix_RGB_to_RGB(
        RGB_COLOURSPACE_BT709, RGB_COLOURSPACE_BT2020
    )
    rec709_to_rec2020_mtx = matrix_RGB_to_RGB(
        RGB_COLOURSPACE_BT2020, RGB_COLOURSPACE_BT709
    )

    print(rec2020_to_rec709_mtx)
    print(rec709_to_rec2020_mtx)
    print(RGB_COLOURSPACE_BT709.primaries.flatten())
    print(RGB_COLOURSPACE_BT709.whitepoint)
    rec709_to_xyz_mtx = normalised_primary_matrix(
        RGB_COLOURSPACE_BT709.primaries.flatten(),
        RGB_COLOURSPACE_BT709.whitepoint
    )
    print(rec709_to_xyz_mtx)
    print(linalg.inv(rec709_to_xyz_mtx))


def get_directX_app_gradient_data(img: np.ndarray):
    num_of_grad_sample = 1025
    num_of_grad_color = 7
    grad_st_pos_h = 1
    grad_st_pos_v_base = 70
    grad_pos_v_offset = 66
    grad_ed_pos_h = grad_st_pos_h + num_of_grad_sample

    gradient_rgb = np.zeros(
        (num_of_grad_color, num_of_grad_sample, 3), dtype=img.dtype)
    for color_idx in range(num_of_grad_color):
        grad_st_pos_v = grad_st_pos_v_base + grad_pos_v_offset * color_idx
        grad_ed_pos_v = grad_st_pos_v + 1
        gradient_rgb[color_idx]\
            = img[grad_st_pos_v:grad_ed_pos_v, grad_st_pos_h:grad_ed_pos_h]

    num_of_rect_color = 4
    rect_st_pos_v = 566
    rect_ed_pos_v = rect_st_pos_v + 1
    rect_st_pox_h_base = 64
    rect_pos_h_offset = 128
    rect_rgb = np.zeros((num_of_rect_color, 3), dtype=img.dtype)
    for color_idx in range(num_of_rect_color):
        rect_st_pox_h = rect_st_pox_h_base + rect_pos_h_offset * color_idx
        rect_ed_pos_h = rect_st_pox_h + 1
        rect_rgb[color_idx]\
            = img[rect_st_pos_v:rect_ed_pos_v, rect_st_pox_h:rect_ed_pos_h]
        
    return gradient_rgb, rect_rgb


def debug_plot_directX_app():
    jxr_file_name = "./Windows_HDR_Capture/DirectX/rgb_10bit.jxr"
    img = imread(jxr_file_name)[..., :3]
    gradient_rgb, rect_rgb = get_directX_app_gradient_data(img=img)
    basename = Path(jxr_file_name).stem

    print(rect_rgb)

    # multi plots
    fig, axes = plt.subplots(7, 1, figsize=(10, 60))
    xx = np.arange(gradient_rgb.shape[1])
    title_list = [
        "White", "Red", "Green", "Blue", "Majenta", "Yellow", "Cyan"
    ]
    for idx in range(7):
        ax1 = axes[idx]
        # yy = (gradient_rgb[idx] ** (1/2.2)) / (100 ** (1/2.2)) * 1023
        yy = gradient_rgb[idx]
        ms = 4
        ax1.plot(xx, yy[..., 0], '-', ms=ms, color=pu.RED, label="R")
        ax1.plot(xx, yy[..., 1], '-', ms=ms, color=pu.GREEN, label="G")
        ax1.plot(xx, yy[..., 2], '-', ms=ms, color=pu.BLUE, label="B")
        # ax1.set_xticks([x * 128 for x in range(8)] + [1023])
        # ax1.set_yticks([x * 256 for x in range(4)] + [1023])
        # ax1.set_xlim([-10, 1033])
        # ax1.set_ylim([-0.8, 0.8])
        ax1.set_title(f'{basename} - {title_list[idx]} Patch')
        ax1.grid(True)
        # pu.log_sacle_settings_x_linear_y_log(ax=ax1)
        ax1.legend(loc='upper left')
    plt.tight_layout()
    save_fname = f"./debug_img/debug_{basename}.png"
    print(save_fname)
    plt.savefig(save_fname, dpi=100)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # plot_captured_hdr_tp()
    # conv_hdr_tp_from_sc_rgb_to_rec2100_pq()
    # plot_tp_7colors(jsr_file="./Windows_HDR_Capture/bak/600.jxr")
    # plot_concat_debug_player_result()
    # check_half_float_error()
    # plot_rec2020_10bit_wrgbmyc_ramp_data_all()
    # plot_rec709_10bit_wrgbmyc_ramp_data_all()
    # debug_plot_check_raw()
    # debug_plot_check_after_conv()
    # debug_check_srgb_rgbw()
    # debug_output_matrix()
    debug_plot_directX_app()
