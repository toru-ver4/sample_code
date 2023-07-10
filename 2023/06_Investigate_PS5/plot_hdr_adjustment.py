# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
import subprocess

# import third-party libraries
import numpy as np

# import my libraries
import plot_utility as pu
import transfer_functions as tf
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2023 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def read_adj_data():
    fname = "./hdr_adjustment.csv"
    data = np.loadtxt(fname=fname, delimiter=',')

    return data


def plot_hdr_adj_on_st2084():
    data = read_adj_data()
    x_high = data[..., 0]
    y_high = data[..., 1]
    x_low = data[..., 0][:16]
    y_low = data[..., 3][:16]
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Adjust HDR (ST2084 scale)",
        graph_title_size=None,
        xlabel="Adjustment Index",
        ylabel="ST2084 Code Value (10-bit)",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=[4 * x for x in range(9)],
        ytick=[128 * x for x in range(8)] + [1023],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(x_high, y_high, '-o', label="Highlight")
    ax1.plot(x_low, y_low, '-o', label="Shadow")
    pu.show_and_save(
        fig=fig, legend_loc='upper left',
        save_fname="./img/adj_plot_st2084.png")


def plot_hdr_adj_on_inear():
    data = read_adj_data()
    x_high = data[..., 0]
    y_high = data[..., 1]
    y_high_l = tf.eotf_to_luminance(y_high/1023, tf.ST2084)
    x_low = data[..., 0][:16]
    y_low = data[..., 3][:16]
    y_low_l = tf.eotf_to_luminance(y_low/1023, tf.ST2084)
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Adjust HDR (Linear scale)",
        graph_title_size=None,
        xlabel="Adjustment Index",
        ylabel="Luminance [nits]",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=[4 * x for x in range(9)],
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    pu.log_sacle_settings_x_linear_y_log(
        ax=ax1, alpha_major=0.6)
    ax1.plot(x_high, y_high_l, '-o', label="Highlight")
    ax1.plot(x_low, y_low_l, '-o', label="Shadow")
    pu.show_and_save(
        fig=fig, legend_loc='upper left',
        save_fname="./img/adj_plot_linear.png")


def actual_plot():
    x_high = np.arange(32)
    x_low = np.arange(16)
    y_high_min = np.round(tf.oetf_from_luminance(100, tf.ST2084) * 1023)
    y_high_max = np.round(tf.oetf_from_luminance(10000, tf.ST2084) * 1023)
    y_low_min = np.round(tf.oetf_from_luminance(0, tf.ST2084) * 1023)
    y_low_max = np.round(tf.oetf_from_luminance(1, tf.ST2084) * 1023)
    y_high = np.linspace(y_high_min, y_high_max, len(x_high))
    y_low = np.linspace(y_low_min, y_low_max, len(x_low))

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Adjust HDR (ST2084 scale)",
        graph_title_size=None,
        xlabel="Adjustment Index",
        ylabel="ST2084 Code Value (10-bit)",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=[4 * x for x in range(9)],
        ytick=[128 * x for x in range(8)] + [1023],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(x_high, y_high, '-o', label="Highlight")
    ax1.plot(x_low, y_low, '-o', label="Shadow")
    pu.show_and_save(
        fig=fig, legend_loc='upper left',
        save_fname="./img/adj_plot_st2084_correct.png")

    y_high_l = tf.eotf_to_luminance(y_high/1023, tf.ST2084)
    y_low_l = tf.eotf_to_luminance(y_low/1023, tf.ST2084)
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Adjust HDR (Linear scale)",
        graph_title_size=None,
        xlabel="Adjustment Index",
        ylabel="Luminance [nits]",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=[4 * x for x in range(9)],
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    pu.log_sacle_settings_x_linear_y_log(
        ax=ax1, alpha_major=0.6)
    ax1.plot(x_high, y_high_l, '-o', label="Highlight")
    ax1.plot(x_low, y_low_l, '-o', label="Shadow")
    pu.show_and_save(
        fig=fig, legend_loc='upper left',
        save_fname="./img/adj_plot_linear_correct.png")

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=None,
        graph_title_size=None,
        xlabel="Adjustment Index",
        ylabel="Luminance [nits]",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=[4 * x for x in range(9)],
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    pu.log_sacle_settings_x_linear_y_log(
        ax=ax1, alpha_major=0.6)
    ax1.plot(x_high, y_high_l, '-o', label="Estimated Highlight Luminance")
    # ax1.plot(x_low, y_low_l, '-o', label="Shadow")
    pu.show_and_save(
        fig=fig, legend_loc='upper left',
        save_fname="./img/adj_plot_linear_correct_high_only.png")


def create_white():
    img = np.ones((2160, 3840, 3)) * 1
    tpg.img_wirte_float_as_16bit_int("./img/white.png", img)


def convert_container_mp4_to_webm():
    src_path = "./img/tone_mapping_checker.mp4"
    dst_path = "./img/tone_mapping_checker.webm"
    cmd = "ffmpeg"
    ops = [
        '-color_primaries', 'bt2020', '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc', '-color_range', 'tv',
        '-i', src_path, '-c:v', 'libvpx-vp9', '-crf', '0',
        '-pix_fmt', 'yuv420p10le',
        '-color_primaries', 'bt2020', '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc', '-color_range', 'tv',
        dst_path, '-y'
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def convert_container_webm_to_mp4():
    src_path = "./img/tone_mapping_checker.webm"
    dst_path = "./img/tone_mapping_checker_verify.mp4"
    cmd = "ffmpeg"
    ops = [
        '-color_primaries', 'bt2020', '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc', '-color_range', 'tv',
        '-i', src_path, '-c:v', 'copy', '-c:a', 'copy',
        '-color_primaries', 'bt2020', '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc', '-color_range', 'tv',
        dst_path, '-y'
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def debug_printf_st2084():
    l_1 = 100
    l_1_2084 = tf.oetf_from_luminance(l_1, tf.ST2084)
    print(l_1_2084)

    cv = 144 / 1023
    ll = tf.eotf_to_luminance(cv, tf.ST2084)
    print(ll)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # plot_hdr_adj_on_st2084()
    # plot_hdr_adj_on_inear()
    # actual_plot()
    # create_white()
    # convert_container_mp4_to_webm()
    # convert_container_webm_to_mp4(
    debug_printf_st2084()
