# -*- coding: utf-8 -*-
"""

==========

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour.algebra import vector_dot
from colour.io import read_image

# import my libraries
import test_pattern_generator2 as tpg
from create_10bit_ramp_tp import\
    calc_ramp_pattern_block_center_pos_with_color_idx,\
    TP_WIDTH, TP_BLOCK_SIZE, calc_rgb_to_rgb_matrix
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


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_plot_different_bit_depth(bit_depth=16)
    debug_plot_different_bit_depth(bit_depth=32)
    debug_plot_different_bit_depth(bit_depth=64)
    # debug_matrix_error()
    # debug_1018cv()
