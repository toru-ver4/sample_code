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
    red = [166, -12.45313]


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

    debug_plot_captured_three_tp(
        fname="./debug/capture/TP_Rec709_2020_17x17x17_Edge.png")
    debug_plot_captured_three_tp(
        fname="./debug/capture/TP_Rec709_2020_17x17x17_MPC-BE.png")
