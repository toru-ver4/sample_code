# -*- coding: utf-8 -*-
"""

==========

"""

# import standard libraries
import os
from pathlib import Path

# import third-party libraries
import numpy as np
from colour.algebra import vector_dot
from colour.io import read_image
from colour.utilities import tstack
from colour.models import eotf_ST2084, eotf_inverse_ST2084
from colour import normalised_primary_matrix
from scipy import linalg
import matplotlib.pyplot as plt

# import my libraries
import test_pattern_generator2 as tpg
from create_10bit_ramp_tp import\
    calc_ramp_pattern_block_center_pos_with_color_idx, \
    TP_WIDTH, TP_BLOCK_SIZE, TP_BLOCK_HEIGHT, calc_rgb_to_rgb_matrix, \
    TP_FILE_NAME, get_gradient_tp_ref_value
from simulate_windows_signal_processing\
    import simulate_windows_signal_processing

import plot_utility as pu
import transfer_functions as tf
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def conv_rgb_to_rgb(img, src_cs_name, dst_cs_name):
    mtx = calc_rgb_to_rgb_matrix(
        src_cs_name=src_cs_name, dst_cs_name=dst_cs_name)
    dst_img = vector_dot(mtx, img)

    return dst_img


def get_709_on_2020_rgb_from_tp(img):
    num_of_sample = 1024
    num_of_color = 7
    read_data = np.zeros((num_of_sample, num_of_color, 3))

    for s_idx in range(num_of_sample):
        for c_idx in range(num_of_color):
            pos = calc_ramp_pattern_block_center_pos_with_color_idx(
                code_value=s_idx, width=TP_WIDTH, block_size=TP_BLOCK_SIZE,
                color_kind_idx=c_idx
            )
            read_data[s_idx, c_idx] = img[pos[1], pos[0]]

    return read_data


def plot_data_one_color(x, y, label, color=pu.RED, fname_suffix=""):
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
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(x, y, '-o', label=label, color=color)
    pu.show_and_save(
        fig=fig, legend_loc='upper left',
        save_fname=f"./debug/plot_tp_{label}_{fname_suffix}.png")


def plot_data_three_color(x, y3, suffix=""):
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 10),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Title",
        graph_title_size=None,
        xlabel="X Axis Label",
        ylabel="Y Axis Label",
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
    ax1.plot(x, y3[..., 0], '-o', label="R", color=pu.RED)
    ax1.plot(x, y3[..., 1], '-o', label="G", color=pu.GREEN)
    ax1.plot(x, y3[..., 2], '-o', label="B", color=pu.BLUE)
    pu.show_and_save(
        fig=fig, legend_loc='upper left',
        save_fname=f"./debug/plot_tp_{suffix}.png")


def debug_plot_different_bit_depth(bit_depth=16):
    if bit_depth == 16:
        img_709_on_2020 = tpg.img_read_as_float(
            "./debug/src_tp/tp_10bit_ramp_wrgbmyc_rec709.png")
    elif bit_depth == 32:
        img_709_on_2020 = read_image(
            "./debug/src_tp/tp_10bit_ramp_wrgbmyc_rec709.exr")
    elif bit_depth == 64:
        img_709_on_2020 = np.load(
            "./debug/src_tp/tp_10bit_ramp_wrgbmyc_rec709.npy")

    all_plot_suffix = f"after_709_{bit_depth}-bit"
    fname_suffix = f"709_linear_{bit_depth}-bit.png"

    tp_rgb = get_709_on_2020_rgb_from_tp(img=img_709_on_2020)
    code_value_list = [
        1010, 1011, 1012, 1013, 1014, 1015, 1016,
        1017, 1018, 1019, 1020, 1021, 1022, 1023
    ]
    color_idx = 2  # green
    rgb_709_on_2020 = np.zeros((len(code_value_list), 3))
    for idx, code_value in enumerate(code_value_list):
        rgb_709_on_2020[idx] = tp_rgb[code_value, color_idx]

    print(rgb_709_on_2020)
    
    rgb_709_on_2020_linear = tf.eotf_to_luminance(rgb_709_on_2020, tf.ST2084)
    rgb_709_linear = conv_rgb_to_rgb(
        img=rgb_709_on_2020_linear, src_cs_name=cs.BT2020, dst_cs_name=cs.BT709)
    rgb_709_pq = tf.oetf_from_luminance(
        np.clip(rgb_709_linear, 0.0, 10000), tf.ST2084) * 1023
    
    print(rgb_709_pq)

    plot_data_three_color(
        x=code_value_list, y3=rgb_709_pq, suffix=all_plot_suffix)

    plot_data_one_color(
        x=code_value_list, y=rgb_709_linear[..., 0], label="R", color=pu.RED,
        fname_suffix=fname_suffix)
    plot_data_one_color(
        x=code_value_list, y=rgb_709_linear[..., 1], label="G", color=pu.GREEN,
        fname_suffix=fname_suffix)
    plot_data_one_color(
        x=code_value_list, y=rgb_709_linear[..., 2], label="B", color=pu.BLUE,
        fname_suffix=fname_suffix)
    

def debug_matrix_error():
    mtx_709_to_2020 = calc_rgb_to_rgb_matrix(
        src_cs_name=cs.BT709, dst_cs_name=cs.BT2020
    )
    mtx_2020_to_709 = calc_rgb_to_rgb_matrix(
        src_cs_name=cs.BT2020, dst_cs_name=cs.BT709
    )
    dot_mtx = mtx_2020_to_709.dot(mtx_709_to_2020)
    print(dot_mtx)


def debug_1018cv():
    img_709_on_2020 = tpg.img_read_as_float(
        "./debug/src_tp/tp_10bit_ramp_wrgbmyc_rec709.png")
    tp_rgb = get_709_on_2020_rgb_from_tp(img=img_709_on_2020)
    code_value = 1018
    color_idx = 2  # green


def apply_mtx(mtx, rgb, calc_dtype=np.float64):
    return np.einsum(
        "...ij,...j->...i",
        mtx.astype(calc_dtype),
        rgb.astype(calc_dtype),
        dtype=calc_dtype
    )


def calculate_rgb_to_rgb_matrix(
        src_primary_xy, dst_primary_xy,
        src_white=[0.3127, 0.3290], dst_white=[0.3127, 0.3290],
        calc_dtype=np.float64):
    """
    Examples
    --------
    >>> calc_rgb_to_rgb_matrix(src_cs_name=cs.BT709, dst_cs_name=cs.BT2020)
    [[ 0.6274039   0.32928304  0.04331307]
     [ 0.06909729  0.9195404   0.01136232]
     [ 0.01639144  0.08801331  0.89559525]]    
    """
    npm_src = normalised_primary_matrix(
        primaries=src_primary_xy, whitepoint=src_white)
    npm_dst = normalised_primary_matrix(
        primaries=dst_primary_xy, whitepoint=dst_white)
    npm_dst_inv = linalg.inv(npm_dst)

    conv_mtx = npm_dst_inv.dot(npm_src)

    return conv_mtx.astype(calc_dtype)


def scrgb_half_float_error_simulation(calc_dtype=np.float64):
    x = np.arange(0, 1024, 1)
    x = tstack([x, x, x])
    color_mask_list = np.array([
        [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 0, 1], [1, 1, 0], [0, 1, 1]
    ])
    yy = []
    for color_mask in color_mask_list:
        yy_temp = x * color_mask
        yy.append(yy_temp)

    rec2020_primary_xy = [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]]
    rec709_primary_xy = [[0.640, 0.330], [0.300, 0.600], [0.150, 0.060]]
    rec2020_to_rec709_mtx = calculate_rgb_to_rgb_matrix(
        src_primary_xy=rec2020_primary_xy, dst_primary_xy=rec709_primary_xy, calc_dtype=calc_dtype)
    rec709_to_rec2020_mtx = calculate_rgb_to_rgb_matrix(
        src_primary_xy=rec709_primary_xy, dst_primary_xy=rec2020_primary_xy, calc_dtype=calc_dtype)

    rgb_2020_st2084 = np.array(yy, dtype=np.int16)
    rgb_2020_linear = eotf_ST2084(rgb_2020_st2084 / 1023.0).astype(calc_dtype)
    rgb_709_linear = apply_mtx(
        mtx=rec2020_to_rec709_mtx, rgb=rgb_2020_linear, calc_dtype=calc_dtype)
    rgb_2020_linear_2 = apply_mtx(
        mtx=rec709_to_rec2020_mtx, rgb=rgb_709_linear, calc_dtype=calc_dtype)
    rgb_2020_st2084_2 = np.round(eotf_inverse_ST2084(np.clip(rgb_2020_linear_2, 0.0, 10000)) * 1023)\
        .astype(np.int16)

    diff = np.abs(rgb_2020_st2084 - rgb_2020_st2084_2)
    print(diff)

    return rgb_2020_st2084_2


def dirext_x_app_ng_simulation(calc_dtype=np.float64):
    x = np.arange(1023, 1024, 1)
    x = tstack([x, x, x])
    color_mask_list = np.array([
        [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 0, 1], [1, 1, 0], [0, 1, 1]
    ])
    yy = []
    for color_mask in color_mask_list:
        yy_temp = x * color_mask
        yy.append(yy_temp)

    rec2020_primary_xy = [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]]
    rec709_primary_xy = [[0.640, 0.330], [0.300, 0.600], [0.150, 0.060]]
    rec2020_to_rec709_mtx = calculate_rgb_to_rgb_matrix(
        src_primary_xy=rec2020_primary_xy, dst_primary_xy=rec709_primary_xy, calc_dtype=np.float32)
    rec709_to_rec2020_mtx = calculate_rgb_to_rgb_matrix(
        src_primary_xy=rec709_primary_xy, dst_primary_xy=rec2020_primary_xy, calc_dtype=calc_dtype)

    rgb_2020_st2084 = np.array(yy, dtype=np.int16)
    rgb_2020_linear = eotf_ST2084(rgb_2020_st2084 / 1023.0).astype(np.float32)
    rgb_709_linear = apply_mtx(
        mtx=rec2020_to_rec709_mtx, rgb=rgb_2020_linear, calc_dtype=np.float32).astype(calc_dtype)
    rgb_2020_linear_2 = apply_mtx(
        mtx=rec709_to_rec2020_mtx, rgb=rgb_709_linear, calc_dtype=calc_dtype)
    rgb_2020_st2084_2 = np.round(eotf_inverse_ST2084(np.clip(rgb_2020_linear_2, 0.0, 10000)) * 1023)\
        .astype(np.int16)

    diff = np.abs(rgb_2020_st2084 - rgb_2020_st2084_2)
    print(rgb_2020_st2084_2)

    return rgb_2020_st2084_2


def plot_simulated_data(data):
    fig, axes = plt.subplots(7, 1, figsize=(8, 20))
    xx = np.arange(1024)
    title_list = [
        "White", "Red", "Green", "Blue", "Majenta", "Yellow", "Cyan"
    ]

    for idx in range(7):
        ax1 = axes[idx]
        yy = data[idx]
        ms = 4
        ax1.plot(xx, yy[..., 0], '-o', ms=ms, color=pu.RED, label="R")
        ax1.plot(xx, yy[..., 1], '-o', ms=ms, color=pu.GREEN, label="G")
        ax1.plot(xx, yy[..., 2], '-o', ms=ms, color=pu.BLUE, label="B")
        ax1.set_xticks([x * 128 for x in range(8)] + [1023])
        ax1.set_yticks([x * 256 for x in range(4)] + [1023])
        ax1.set_xlim([-10, 1033])
        ax1.set_ylim([-20, 1043])
        ax1.set_title(f'{title_list[idx]} - np.float16')
        ax1.grid(True)
        ax1.legend(loc='upper left')

    plt.tight_layout()
    save_fname = f"./debug/simulation.png"
    print(save_fname)
    plt.savefig(save_fname, dpi=100)


def get_app_name_from_fname(fname):
    fname_base = Path(fname).stem
    app_name = fname_base.rsplit("_", 1)[-1]

    return app_name


def float_to_int10(x):
    return np.round(x * 1023).astype(np.int16)


def debug_plot_captured_rec2020_within_three_tp(
        captured_img, base_fname, app_name):
    cap_2020_img = captured_img[TP_BLOCK_HEIGHT:TP_BLOCK_HEIGHT*2]
    cap_rgb = get_gradient_tp_ref_value(
        tp_img=cap_2020_img, width=TP_WIDTH, block_size=TP_BLOCK_SIZE)
    cap_rgb_int10 = float_to_int10(cap_rgb)

    ref_img = tpg.img_read_as_float(TP_FILE_NAME)
    ref_2020_img = ref_img[TP_BLOCK_HEIGHT:TP_BLOCK_HEIGHT*2]
    ref_rgb = get_gradient_tp_ref_value(
        tp_img=ref_2020_img, width=TP_WIDTH, block_size=TP_BLOCK_SIZE)
    ref_rgb_int10 = float_to_int10(ref_rgb)

    #################################################
    # NORMAL PLOT
    #################################################
    fig, axes = plt.subplots(7, 1, figsize=(8, 20))
    xx = np.arange(1024).astype(np.uint16)
    title_list = [
        "White", "Red", "Green", "Blue", "Majenta", "Yellow", "Cyan"
    ]

    for idx in range(7):
        ax1 = axes[idx]
        yy = cap_rgb_int10[idx]
        y_ref = ref_rgb_int10[idx]
        ms = 4
        ax1.plot(xx, yy[..., 0], '-', ms=ms, color=pu.RED, label="R")
        ax1.plot(xx, yy[..., 1], '-', ms=ms, color=pu.GREEN, label="G")
        ax1.plot(xx, yy[..., 2], '-', ms=ms, color=pu.BLUE, label="B")
        ax1.plot(xx, y_ref[..., 0], '--', ms=ms, color=pu.MAJENTA, label="R_Ref")
        ax1.plot(xx, y_ref[..., 1], '--', ms=ms, color=pu.YELLOW, label="G_Ref")
        ax1.plot(xx, y_ref[..., 2], '--', ms=ms, color=pu.SKY, label="B_Ref")
        ax1.set_xticks([x * 128 for x in range(8)] + [1023])
        ax1.set_yticks([x * 256 for x in range(4)] + [1023])
        ax1.set_xlim([-10, 1033])
        ax1.set_ylim([-20, 1043])
        ax1.set_title(f'{app_name} - {title_list[idx]} Patch')
        ax1.grid(True)
        ax1.legend(loc='upper left')

    plt.tight_layout()

    save_fname = f"./debug/plot/{base_fname}_Rec2020.png"
    print(save_fname)
    plt.savefig(save_fname, dpi=100)

    #################################################
    # DIFF PLOT
    #################################################
    fig, axes = plt.subplots(7, 1, figsize=(8, 20))
    xx = np.arange(1024).astype(np.uint16)
    title_list = [
        "White", "Red", "Green", "Blue", "Majenta", "Yellow", "Cyan"
    ]

    for idx in range(7):
        ax1 = axes[idx]
        yy = cap_rgb_int10[idx]
        y_ref = ref_rgb_int10[idx]
        ms = 4
        ax1.plot(xx, y_ref[..., 0] - yy[..., 0], '-', ms=ms, color=pu.RED, label="R_diff")
        ax1.plot(xx, y_ref[..., 1] - yy[..., 1], '-', ms=ms, color=pu.GREEN, label="G_diff")
        ax1.plot(xx, y_ref[..., 2] - yy[..., 2], '-', ms=ms, color=pu.BLUE, label="B_diff")
        ax1.set_xticks([x * 128 for x in range(8)] + [1023])
        ax1.set_xlim([-10, 1033])
        ax1.set_ylim([-200, 200])
        ax1.set_title(f'{app_name} - {title_list[idx]} Patch')
        ax1.grid(True)
        ax1.legend(loc='upper left')

    plt.tight_layout()

    save_fname = f"./debug/plot/{base_fname}_Rec2020_diff.png"
    print(save_fname)
    plt.savefig(save_fname, dpi=100)


def debug_plot_captured_rec709_within_three_tp(
        captured_img, base_fname, app_name):
    cap_709_img = captured_img[0:TP_BLOCK_HEIGHT]
    cap_rgb = get_gradient_tp_ref_value(
        tp_img=cap_709_img, width=TP_WIDTH, block_size=TP_BLOCK_SIZE)
    cap_rgb_int10 = float_to_int10(cap_rgb)

    ref_img = tpg.img_read_as_float(TP_FILE_NAME)
    ref_709_img = ref_img[0:TP_BLOCK_HEIGHT]
    ref_rgb = get_gradient_tp_ref_value(
        tp_img=ref_709_img, width=TP_WIDTH, block_size=TP_BLOCK_SIZE)
    ref_rgb_int10 = float_to_int10(ref_rgb)

    #################################################
    # NORMAL PLOT
    #################################################
    fig, axes = plt.subplots(7, 1, figsize=(8, 20))
    xx = np.arange(1024).astype(np.uint16)
    title_list = [
        "White", "Red", "Green", "Blue", "Majenta", "Yellow", "Cyan"
    ]

    for idx in range(7):
        ax1 = axes[idx]
        yy = cap_rgb_int10[idx]
        y_ref = ref_rgb_int10[idx]
        ms = 4
        ax1.plot(xx, yy[..., 0], '-', ms=ms, color=pu.RED, label="R")
        ax1.plot(xx, yy[..., 1], '-', ms=ms, color=pu.GREEN, label="G")
        ax1.plot(xx, yy[..., 2], '-', ms=ms, color=pu.BLUE, label="B")
        ax1.plot(xx, y_ref[..., 0], '--', ms=ms, color=pu.MAJENTA, label="R_Ref")
        ax1.plot(xx, y_ref[..., 1], '--', ms=ms, color=pu.YELLOW, label="G_Ref")
        ax1.plot(xx, y_ref[..., 2], '--', ms=ms, color=pu.SKY, label="B_Ref")
        ax1.set_xticks([x * 128 for x in range(8)] + [1023])
        ax1.set_yticks([x * 256 for x in range(4)] + [1023])
        ax1.set_xlim([-10, 1033])
        ax1.set_ylim([-20, 1043])
        ax1.set_title(f'{app_name} - {title_list[idx]} Patch')
        ax1.grid(True)
        ax1.legend(loc='upper left')

    plt.tight_layout()

    save_fname = f"./debug/plot/{base_fname}_Rec709.png"
    print(save_fname)
    plt.savefig(save_fname, dpi=100)

    #################################################
    # DIFF PLOT
    #################################################
    fig, axes = plt.subplots(7, 1, figsize=(8, 20))
    xx = np.arange(1024).astype(np.uint16)
    title_list = [
        "White", "Red", "Green", "Blue", "Majenta", "Yellow", "Cyan"
    ]

    for idx in range(7):
        ax1 = axes[idx]
        yy = cap_rgb_int10[idx]
        y_ref = ref_rgb_int10[idx]
        ms = 4
        ax1.plot(xx, y_ref[..., 0] - yy[..., 0], '-', ms=ms, color=pu.RED, label="R_diff")
        ax1.plot(xx, y_ref[..., 1] - yy[..., 1], '-', ms=ms, color=pu.GREEN, label="G_diff")
        ax1.plot(xx, y_ref[..., 2] - yy[..., 2], '-', ms=ms, color=pu.BLUE, label="B_diff")
        ax1.set_xticks([x * 128 for x in range(8)] + [1023])
        ax1.set_xlim([-10, 1033])
        ax1.set_ylim([-20, 1043])
        ax1.set_ylim([-200, 200])
        ax1.set_title(f'{app_name} - {title_list[idx]} Patch')
        ax1.grid(True)
        ax1.legend(loc='upper left')

    plt.tight_layout()

    save_fname = f"./debug/plot/{base_fname}_Rec709_diff.png"
    print(save_fname)
    plt.savefig(save_fname, dpi=100)


def debug_plot_captured_three_tp(fname: str):
    img = tpg.img_read_as_float(fname)
    base_fname = Path(fname).stem
    app_name = get_app_name_from_fname(fname=fname)

    debug_plot_captured_rec2020_within_three_tp(
        captured_img=img, base_fname=base_fname, app_name=app_name)
    debug_plot_captured_rec709_within_three_tp(
        captured_img=img, base_fname=base_fname, app_name=app_name)


def calc_matrix_based_on_DWM():
    rr = np.array([166, -12.45313, -1.81445]) / 100.0
    gg = np.array([-58.75, 113.25, -10.05469]) / 100.0
    bb = np.array([-7.28516, -0.83447, 111.81250]) / 100.0

    mtx = np.zeros((3, 3), dtype=np.float64)
    mtx[0, 0] = rr[0]
    mtx[0, 1] = gg[0]
    mtx[0, 2] = bb[0]
    mtx[1, 0] = rr[1]
    mtx[1, 1] = gg[1]
    mtx[1, 2] = bb[1]
    mtx[2, 0] = rr[2]
    mtx[2, 1] = gg[2]
    mtx[2, 2] = bb[2]

    mtx = mtx.astype(np.float16)

    print(mtx)


def calc_half_float_inv_rec709_to_rec2020_mtx():
    rec709_to_rec2020_mtx = calc_rgb_to_rgb_matrix(
        src_cs_name=cs.BT709, dst_cs_name=cs.BT2020)
    rec709_to_rec2020_mtx_half = rec709_to_rec2020_mtx.astype(np.float16)
    rec2020_to_rec709_mtx = np.linalg.inv(rec709_to_rec2020_mtx_half.astype(np.float32))
    rec2020_to_rec709_mtx_half = rec2020_to_rec709_mtx.astype(np.float16)
    print(rec2020_to_rec709_mtx_half)
    dot_mtx = rec709_to_rec2020_mtx_half.dot(rec2020_to_rec709_mtx_half)
    print(dot_mtx)


def plot_tp_10bit_green_high_luminance_hdmi():
    fname = "../05_Correct_Windows_HDR_Output/Windows_HDR_Capture/gain_1.0_10-bit/TP_Rec2020_10-bit_Edge_hdmi.png"
    img = tpg.img_read_as_float(fname)
    rgb_data = get_gradient_tp_ref_value(
        tp_img=img, width=3840, block_size=32)
    gg = rgb_data[2] * 1023

    x = np.arange(768, 1024)
    print(x)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(14, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Edge_hdmi - Green",
        graph_title_size=None,
        xlabel="Target Code Value (10-bit)",
        ylabel="Measured Code Value (10-bit)",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=[768 + x * 32 for x in range(8)] + [1023],
        ytick=[x * 128 for x in range(8)] + [1023],
        xtick_size=None, ytick_size=None,
        linewidth=2,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(x, gg[x, 0], '-o', color=pu.RED, label="R")
    ax1.plot(x, gg[x, 1], '-o', color=pu.GREEN, label="G")
    ax1.plot(x, gg[x, 2], '-o', color=pu.BLUE, label="B")
    pu.show_and_save(
        fig=fig, legend_loc='upper left', show=False,
        save_fname="./debug/plot/TP_2020_Edge_HDMI_Green_Magnified.png")


def create_tp_corrdinate_and_ref_value_csv():
    def core_func_rec709_rec2020(
            ref_tp_fname, out_csv_name, v_offset_for_type, color_name_list,
            num_of_color, num_of_cv, csv_header):
        coordinate_info = np.zeros(
            (num_of_color, num_of_cv, 2), dtype=np.uint16)
        reference_cv = np.zeros((num_of_color, num_of_cv, 3), dtype=np.uint16)
        ref_img = tpg.img_read_as_float(ref_tp_fname)
        ref_img = np.round(ref_img * 1023).astype(np.uint16)

        for c_idx in range(num_of_color):
            for cv in range(num_of_cv):
                center_pos = calc_ramp_pattern_block_center_pos_with_color_idx(
                    code_value=cv, width=width, block_size=block_size,
                    color_kind_idx=c_idx
                )
                center_pos[1] = center_pos[1] + v_offset_for_type
                coordinate_info[c_idx, cv] = center_pos
                reference_cv[c_idx, cv] = ref_img[center_pos[1], center_pos[0]]

        with open(out_csv_name, 'wt') as f:
            buf = ""
            buf += csv_header
            for c_idx in range(num_of_color):
                color_name = color_name_list[c_idx]
                for cv in range(num_of_cv):
                    center_pos = coordinate_info[c_idx, cv]
                    ref_rgb = reference_cv[c_idx, cv]
                    buf += f"{color_name},{cv},"
                    buf += f"{center_pos[0]},{center_pos[1]},"
                    buf += f"{ref_rgb[0]},{ref_rgb[1]},{ref_rgb[2]}\n"

            f.write(buf)

    width = TP_WIDTH
    block_size = TP_BLOCK_SIZE
    num_of_color = 7
    num_of_cv = 1024
    csv_header = "colors,code_value,pos_x,pos_y,ref_R,ref_G,ref_B\n"
    color_name_list =\
        ["white", "red", "green", 'blue', 'magenta', 'cyan', 'yellow']
    ref_tp_fname = "./debug/src_tp/10bit_gradient_tp_709_2020_17x17x17.png"
    v_offset_for_type = 720  # 2160 (px) / 3 (type) = 720 px

    # Rec.709 on Re.2020
    out_csv_name = "./debug/tp_coordinate/tp_coordinate_rec709_on_rec2020.csv"
    core_func_rec709_rec2020(
        ref_tp_fname=ref_tp_fname, color_name_list=color_name_list,
        out_csv_name=out_csv_name, v_offset_for_type=v_offset_for_type * 0,
        num_of_color=num_of_color, num_of_cv=num_of_cv, csv_header=csv_header
    )

    # Rec.2020
    out_csv_name = "./debug/tp_coordinate/tp_coordinate_rec2020.csv"
    core_func_rec709_rec2020(
        ref_tp_fname=ref_tp_fname, color_name_list=color_name_list,
        out_csv_name=out_csv_name, v_offset_for_type=v_offset_for_type * 1,
        num_of_color=num_of_color, num_of_cv=num_of_cv, csv_header=csv_header
    )

    # 17x17x17
    num_of_color = 1
    num_of_cv = 17 ** 3
    csv_header = "type,sequential_index,pos_x,pos_y,ref_R,ref_G,ref_B\n"
    out_csv_name = "./debug/tp_coordinate/tp_coordinate_17x17x17.csv"
    color_name_list = ["17x17x17"]
    core_func_rec709_rec2020(
        ref_tp_fname=ref_tp_fname, color_name_list=color_name_list,
        out_csv_name=out_csv_name, v_offset_for_type=v_offset_for_type * 2,
        num_of_color=num_of_color, num_of_cv=num_of_cv, csv_header=csv_header
    )


def plot_diff_rec709_rec2020_single_diff_base(
        capture_png_fname, color_space="Rec.709"):
    basename = Path(capture_png_fname).stem
    title_base = basename.replace("TP_Rec709_2020_17x17x17_HDMI_", "")
    img_all = tpg.img_read_as_float(capture_png_fname)
    img_all = np.round(img_all * 1023).astype(np.int16)
    if color_space == 'Rec.709':
        csv_fname = "./debug/tp_coordinate/tp_coordinate_rec709_on_rec2020.csv"
    elif color_space == "Rec.2020":
        csv_fname = "./debug/tp_coordinate/tp_coordinate_rec2020.csv"
    else:
        raise ValueError(f'color_space: {color_space} is invalid parameter')
    pos_and_ref_rgb_list = np.loadtxt(
        fname=csv_fname, delimiter=',', skiprows=1, usecols=np.arange(1, 7),
        dtype=np.int16)
    pos_and_ref_rgb_list = pos_and_ref_rgb_list.reshape(7, 1024, 6)
    cv_list = pos_and_ref_rgb_list[..., 0]
    pos_list = pos_and_ref_rgb_list[..., 1:3]
    ref_rgb_list = pos_and_ref_rgb_list[..., 3:6]

    captured_rgb_list = img_all[pos_list[..., 1], pos_list[..., 0]]
    diff_rgb = captured_rgb_list - ref_rgb_list

    fig, axes = plt.subplots(7, 1, figsize=(8, 20))
    title_list = [
        "White", "Red", "Green", "Blue", "Majenta", "Yellow", "Cyan"
    ]
    num_of_color = len(title_list)

    for idx in range(num_of_color):
        ax1 = axes[idx]
        xx = cv_list[idx]
        yy = diff_rgb[idx]
        ms = 4
        ax1.plot(xx, yy[..., 0], '-', ms=ms, color=pu.RED, label="R")
        ax1.plot(xx, yy[..., 1], '-', ms=ms, color=pu.GREEN, label="G")
        ax1.plot(xx, yy[..., 2], '-', ms=ms, color=pu.BLUE, label="B")
        ax1.set_xticks([x * 128 for x in range(8)] + [1023])
        ax1.set_yticks([-100 + x * 50 for x in range(9)])
        ax1.set_xlim([-10, 1033])
        ax1.set_ylim([-100, 300])
        ax1.set_xlabel('Target Code Value (10-bit)')
        ax1.set_ylabel('Difference (10-bit)')
        ax1.set_title(f'{title_base} - {title_list[idx]}')
        ax1.grid(True)
        ax1.legend(loc='upper left')

    plt.tight_layout()

    save_fname = f"./debug/plot/diff_{basename}_{color_space}.png"
    print(save_fname)
    plt.savefig(save_fname, dpi=100)

    return save_fname


def plot_diff_rec709_rec2020_single_reference_base(
        capture_png_fname, color_space="Rec.709"):
    basename = Path(capture_png_fname).stem
    title_base = basename.replace("TP_Rec709_2020_17x17x17_HDMI_", "")
    img_all = tpg.img_read_as_float(capture_png_fname)
    img_all = np.round(img_all * 1023).astype(np.int16)
    if color_space == 'Rec.709':
        csv_fname = "./debug/tp_coordinate/tp_coordinate_rec709_on_rec2020.csv"
    elif color_space == "Rec.2020":
        csv_fname = "./debug/tp_coordinate/tp_coordinate_rec2020.csv"
    else:
        raise ValueError(f'color_space: {color_space} is invalid parameter')
    pos_and_ref_rgb_list = np.loadtxt(
        fname=csv_fname, delimiter=',', skiprows=1, usecols=np.arange(1, 7),
        dtype=np.int16)
    pos_and_ref_rgb_list = pos_and_ref_rgb_list.reshape(7, 1024, 6)
    cv_list = pos_and_ref_rgb_list[..., 0]
    pos_list = pos_and_ref_rgb_list[..., 1:3]
    ref_rgb_list = pos_and_ref_rgb_list[..., 3:6]

    captured_rgb_list = img_all[pos_list[..., 1], pos_list[..., 0]]

    fig, axes = plt.subplots(7, 1, figsize=(8, 20))
    title_list = [
        "White", "Red", "Green", "Blue", "Majenta", "Yellow", "Cyan"
    ]
    num_of_color = len(title_list)

    for idx in range(num_of_color):
        ax1 = axes[idx]
        xx = cv_list[idx]
        yy = captured_rgb_list[idx]
        ref = ref_rgb_list[idx]
        ms = 4
        ax1.plot(xx, yy[..., 0], '-', ms=ms, color=pu.RED, label="R")
        ax1.plot(xx, yy[..., 1], '-', ms=ms, color=pu.GREEN, label="G")
        ax1.plot(xx, yy[..., 2], '-', ms=ms, color=pu.BLUE, label="B")
        ax1.plot(xx, ref[..., 0], '--', ms=ms, color=pu.MAJENTA, alpha=0.5, label="R (Reference)")
        ax1.plot(xx, ref[..., 1], '--', ms=ms, color=pu.YELLOW, alpha=0.5, label="G (Reference)")
        ax1.plot(xx, ref[..., 2], '--', ms=ms, color=pu.SKY, alpha=0.5, label="B (Reference)")
        ax1.set_xticks([x * 128 for x in range(8)] + [1023])
        ax1.set_yticks([x * 256 for x in range(4)] + [1023])
        ax1.set_xlim([-10, 1033])
        ax1.set_ylim([-20, 1043])
        ax1.set_xlabel('Target Code Value (10-bit)')
        ax1.set_ylabel('Measured Code Value (10-bit)')
        ax1.set_title(f'{title_base} - {title_list[idx]}')
        ax1.grid(True)
        ax1.legend(loc='upper left')

    plt.tight_layout()

    save_fname = f"./debug/plot/ref_{basename}_{color_space}.png"
    print(save_fname)
    plt.savefig(save_fname, dpi=100)

    return save_fname


def plot_diff_rec709_rec2020_control():
    def plot_and_concat(capture_png_fname_list, color_space):
        graph_diff_fname_list = []
        graph_ref_fname_list = []
        for capture_png_fname in capture_png_fname_list:
            graph_diff_fname = plot_diff_rec709_rec2020_single_diff_base(
                capture_png_fname=capture_png_fname, color_space=color_space)
            graph_diff_fname_list.append(graph_diff_fname)

            graph_ref_fname = plot_diff_rec709_rec2020_single_reference_base(
                capture_png_fname=capture_png_fname, color_space=color_space)
            graph_ref_fname_list.append(graph_ref_fname)

        img_diff_list = []
        img_ref_list = []
        for graph_diff_fname, graph_ref_fname in\
                zip(graph_diff_fname_list, graph_ref_fname_list):
            img = tpg.img_read(graph_diff_fname)
            img_diff_list.append(img)

            img = tpg.img_read(graph_ref_fname)
            img_ref_list.append(img)

        out_img = np.hstack(img_diff_list)
        concat_fname = "./debug/plot/concat_diff_TP_Rec709_2020_17x17x17"
        concat_fname += f"_{color_space}.png"
        tpg.img_write(concat_fname, out_img)

        out_img = np.hstack(img_ref_list)
        concat_fname = "./debug/plot/concat_ref_TP_Rec709_2020_17x17x17"
        concat_fname += f"_{color_space}.png"
        tpg.img_write(concat_fname, out_img)

    capture_png_fname_list = [
        "./debug/capture/TP_Rec709_2020_17x17x17_HDMI_Edge.png",
        "./debug/capture/TP_Rec709_2020_17x17x17_HDMI_Chrome.png",
        "./debug/capture/TP_Rec709_2020_17x17x17_HDMI_MPC-BE.png",
        "./debug/capture/TP_Rec709_2020_17x17x17_HDMI_movies_and_TV.png"
    ]
    plot_and_concat(
        capture_png_fname_list=capture_png_fname_list,
        color_space="Rec.2020")
    plot_and_concat(
        capture_png_fname_list=capture_png_fname_list,
        color_space="Rec.709")


def create_diff_csv_17x17x17_single(capture_png_fname):
    basename = Path(capture_png_fname).stem
    # title_base = basename.replace("TP_Rec709_2020_17x17x17_HDMI_", "")
    img_all = tpg.img_read_as_float(capture_png_fname)
    img_all = np.round(img_all * 1023).astype(np.int16)
    csv_fname = "./debug/tp_coordinate/tp_coordinate_17x17x17.csv"
    pos_and_ref_rgb_list = np.loadtxt(
        fname=csv_fname, delimiter=',', skiprows=1, usecols=np.arange(1, 7),
        dtype=np.int16)
    pos_list = pos_and_ref_rgb_list[..., 1:3]
    ref_rgb_list = pos_and_ref_rgb_list[..., 3:6]
    captured_rgb_list = img_all[pos_list[..., 1], pos_list[..., 0]]
    # normalize_coef = np.max(ref_rgb_list, axis=-1).reshape(-1, 1)
    # normalize_coef[normalize_coef == 0] = 1

    # nor_captured_rgb_list = captured_rgb_list / normalize_coef
    # nor_ref_rgb_list = ref_rgb_list / normalize_coef

    # diff = np.abs(nor_captured_rgb_list - nor_ref_rgb_list)
    # diff_sum = np.sum(diff, axis=-1)

    csv_file_fname = f"./debug/plot/diff_17x17x17_{basename}.csv"
    header = "idx,ref_r,ref_g,reg_b,measure_r,measure_g,measure_b\n"
    with open(csv_file_fname, "wt") as f:
        buf = ""
        buf += header
        for idx in range(17**3):
            buf += f"{idx},"
            buf += f"{ref_rgb_list[idx, 0]},"
            buf += f"{ref_rgb_list[idx, 1]},"
            buf += f"{ref_rgb_list[idx, 2]},"
            buf += f"{captured_rgb_list[idx, 0]},"
            buf += f"{captured_rgb_list[idx, 1]},"
            buf += f"{captured_rgb_list[idx, 2]},"
            buf += "\n"
        f.write(buf)


def create_diff_csv_17x17x17_control():
    capture_png_fname_list = [
        "./debug/capture/TP_Rec709_2020_17x17x17_HDMI_Edge.png",
        "./debug/capture/TP_Rec709_2020_17x17x17_HDMI_Chrome.png",
        "./debug/capture/TP_Rec709_2020_17x17x17_HDMI_MPC-BE.png",
        "./debug/capture/TP_Rec709_2020_17x17x17_HDMI_movies_and_TV.png"
    ]

    for capture_png_fname in capture_png_fname_list:
        create_diff_csv_17x17x17_single(
            capture_png_fname=capture_png_fname
        )
        break


def plot_inverse_st2084():
    min_val = 10 ** (-4)
    max_val = 10000
    num_of_sample = 32
    min_log = np.log2(min_val)
    max_log = np.log2(max_val)
    print(min_log, max_log)

    # Linear Scale
    x = np.arange(10001)
    y = eotf_inverse_ST2084(x) * 1023
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="SMPTE ST 2084 Inverse EOTF",
        graph_title_size=None,
        xlabel="Display Light (cd/m2)",
        ylabel="Code Value (10-bit)",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=[x * 128 for x in range(8)] + [1023],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(x, y, label="SMPTE ST 2084 Inverse EOTF")
    pu.show_and_save(
        fig=fig, legend_loc='upper left', show=True,
        save_fname="./debug/plot/pq_oetf_linear.png")

    # Log Scale
    x = 2 ** (np.linspace(min_log, max_log, num_of_sample))
    y = eotf_inverse_ST2084(x) * 1023
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="SMPTE ST 2084 Inverse EOTF",
        graph_title_size=None,
        xlabel="Display Light (cd/m2)",
        ylabel="Code Value (10-bit)",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=[x * 128 for x in range(8)] + [1023],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    pu.log_sacle_settings_x_log_y_linear(
        ax=ax1, alpha_major=0.6, alpha_minor=0.2)
    ax1.plot(x, y, label="SMPTE ST 2084 Inverse EOTF")
    pu.show_and_save(
        fig=fig, legend_loc='upper left', show=False,
        save_fname="./debug/plot/pq_oetf_log.png")


def plot_simulated_green_only_data():
    dummy1, dummy2, rgb = simulate_windows_signal_processing()

    x = np.arange(1024).astype(np.uint16)
    y = rgb[2]  # green data

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Simulated HDMI Output (Green Gradient)",
        graph_title_size=None,
        xlabel="Target Code Value (10-bit)",
        ylabel="Simulated Code Value (10-bit)",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=[128 * x for x in range(8)] + [1023],
        ytick=[128 * x for x in range(8)] + [1023],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(x, y[..., 0], '-', color=pu.RED, label="R")
    ax1.plot(x, y[..., 1], '-', color=pu.GREEN, label="G")
    ax1.plot(x, y[..., 2], '-', color=pu.BLUE, label="B")
    pu.show_and_save(
        fig=fig, legend_loc='upper left', show=False,
        save_fname="./debug/plot/simulated_data_green_only.png")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # debug_plot_different_bit_depth(bit_depth=16)
    # debug_plot_different_bit_depth(bit_depth=32)
    # debug_plot_different_bit_depth(bit_depth=64)
    # debug_matrix_error()
    # debug_1018cv()

    # data = scrgb_half_float_error_simulation(calc_dtype=np.float16)
    # plot_simulated_data(data=data)

    # dirext_x_app_ng_simulation(calc_dtype=np.float16)

    # DO NOT FORGET TO IMPLEMENT THIS FUNCTION!!!!!!
    # calc_matrix_based_on_DWM()

    # debug_plot_captured_three_tp(
    #     fname="./debug/capture/TP_Rec709_2020_17x17x17_Edge.png")
    # debug_plot_captured_three_tp(
    #     fname="./debug/capture/TP_Rec709_2020_17x17x17_MPC-BE.png")
    # calc_half_float_inv_rec709_to_rec2020_mtx()

    # plot_tp_10bit_green_high_luminance_hdmi()
    # create_tp_corrdinate_and_ref_value_csv()

    # plot_diff_rec709_rec2020_control()
    # create_diff_csv_17x17x17_control()

    # plot_inverse_st2084()

    # print(eotf_ST2084(128/1023))
    # x = np.array([0.01, 1000, 0.1])
    # y = np.round(tf.oetf_from_luminance(x, tf.ST2084) * 1023).astype(np.uint16)
    # print(y)

    plot_simulated_green_only_data()
