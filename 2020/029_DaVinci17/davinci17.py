# -*- coding: utf-8 -*-
"""

==========

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from colour import write_image, read_image

# import my libraries
import test_pattern_generator2 as tpg
import transfer_functions as tf
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def create_ramp():
    x = np.linspace(0, 1, 1920).reshape((1, 1920, 1))

    img = np.ones((1080, 1920, 3))
    img = x * img
    write_image(img, "test_src.tif", bit_depth='uint16')


def create_exr_ramp(min_exposure=-12, max_exposure=12):
    x = np.linspace(0, 1, 1920).reshape((1, 1920, 1))
    y = tpg.shaper_func_log2_to_linear(
        x, min_exposure=min_exposure, max_exposure=max_exposure)

    img = np.ones((1080, 1920, 3)) * y
    fname = f"./img/test_src_exp_{min_exposure}_{max_exposure}.exr"
    write_image(img, fname, bit_depth='float32')


def plot_input_drt():
    # file_list = [
    #     ['./img/test_out_sdr100.tif', 'SDR 100'],
    #     ['./img/test_out_hdr500.tif', 'HDR 500'],
    #     ['./img/test_out_hdr1000.tif', 'HDR 1000'],
    #     ['./img/test_out_hdr2000.tif', 'HDR 2000'],
    #     ['./img/test_out_hdr4000.tif', 'HDR 4000'],
    #     ['./img/test_out_off.tif', 'DRT OFF']
    # ]
    # check_input_drt_test(
    #     file_list=file_list, graph_name="input_drt_spec_01")

    # file_list = [
    #     ['./img/test_out_sdr_er_100-200.tif', 'SDR ER 100/200'],
    #     ['./img/test_out_hdr_er_1000-2000.tif', 'HDR ER 1000/2000'],
    #     ['./img/test_out_hdr_er_1000-4000.tif', 'HDR ER 1000/4000'],
    #     ['./img/test_out_hdr_er_1000-10000.tif', 'HDR ER 1000/10000'],
    #     ['./img/test_out_hdr_er_4000-10000.tif', 'HDR ER 4000/10000'],
    #     ['./img/test_out_off.tif', 'DRT OFF']
    # ]
    # check_input_drt_test(
    #     file_list=file_list, graph_name="input_drt_spec_02")

    check_input_drt_test_sdr_only()


def check_input_drt_test(file_list, graph_name):
    create_ramp()
    x = np.linspace(0, 1, 1920)
    x_luminance = tf.eotf_to_luminance(x, tf.ST2084)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="DaVinci17 Input DRT Characteristics",
        graph_title_size=None,
        xlabel="Input Luminance [cd/m2]",
        ylabel="Output Luminance [cd/m2]",
        axis_label_size=None,
        legend_size=17,
        xlim=[0.009, 15000],
        ylim=[0.009, 15000],
        xtick=None,
        ytick=None,
        xtick_size=None,
        ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None,
        return_figure=True)
    pu.log_scale_settings(ax1, grid_alpha=0.5, bg_color="#E0E0E0")

    for idx in range(len(file_list))[::-1]:
        img = read_image(file_list[idx][0])[0, :, 0]
        label = file_list[idx][1]
        y_luminance = tf.eotf_to_luminance(img, tf.ST2084)
        ax1.plot(x_luminance, y_luminance, label=label)

    plt.legend(loc='upper left')
    fname_full = f"./img/{graph_name}.png"
    plt.savefig(fname_full, bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close(fig)


def check_input_drt_test_sdr_only():
    create_ramp()
    x = np.linspace(0, 1, 1920)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="DaVinci17 Input DRT Characteristics",
        graph_title_size=None,
        xlabel="Input Luminance [cd/m2]",
        ylabel="Output Luminance [cd/m2]",
        axis_label_size=None,
        legend_size=17,
        xlim=[0.009, 15000],
        ylim=[0.009, 15000],
        xtick=None,
        ytick=None,
        xtick_size=None,
        ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None,
        return_figure=True)
    pu.log_scale_settings(ax1, grid_alpha=0.5, bg_color="#E0E0E0")

    img = read_image("./img/test_out_sdr100_on_gm24.tif")[0, :, 0]
    label = "DRT OFF(ST2084 to Gamma2.4 (.tif))"
    x_luminance = tf.eotf_to_luminance(x, tf.ST2084)
    y_luminance = tf.eotf_to_luminance(img, tf.GAMMA24)
    ax1.plot(x_luminance, y_luminance, label=label)

    img = read_image("./img/test_out_sdr100_on_gm24_203nits.tif")[0, :, 0]
    label = "DRT OFF(ST2084 to Gamma2.4 (.tif) 203nits)"
    x_luminance = tf.eotf_to_luminance(x, tf.ST2084)
    y_luminance = tf.eotf_to_luminance(img, tf.GAMMA24)
    ax1.plot(x_luminance, y_luminance, label=label)

    # img = read_image("./img/test_out_sdr100_on_gm24.tif")[0, :, 0]
    # label = 'SDR 100 (ST2084 to Gamma2.4 (.tif))'
    # x_luminance = tf.eotf_to_luminance(x, tf.ST2084)
    # y_luminance = tf.eotf_to_luminance(img, tf.GAMMA24)
    # ax1.plot(x_luminance, y_luminance, label=label)

    # img = read_image("./img/test_out_exp_-12_12_sdr_drt-off_gm24.tif")[0, :, 0]
    # label = "DRT OFF(Gamma2.4 to Gamma2.4 (.tif))"
    # x_luminance = tf.eotf_to_luminance(x, tf.GAMMA24)
    # y_luminance = tf.eotf_to_luminance(img, tf.GAMMA24)
    # ax1.plot(x_luminance, y_luminance, label=label)

    # img = read_image("./img/test_out_exp_-12_12_sdr_drt-off.tif")[0, :, 0]
    # label = "DRT OFF(Linear to Gamma2.4 (.exr))"
    # y_luminance = tf.eotf_to_luminance(img, tf.GAMMA24)
    # x = np.linspace(0, 1, 1920)
    # x_luminance = tpg.shaper_func_log2_to_linear(
    #     x, min_exposure=-12, max_exposure=12)
    # ax1.plot(
    #     x_luminance * 100, y_luminance, '--', color=pu.SKY, label=label)

    plt.legend(loc='upper left')
    fname_full = "./img/input_drt_sdr_only.png"
    plt.savefig(fname_full, bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close(fig)


def check_100nits_code_value_on_st2084():
    code_value = tf.oetf_from_luminance(100, tf.ST2084)
    print(code_value)
    print(code_value * 1023)


def plot_forum_fig1():
    x = np.linspace(0, 1, 1920)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="HDR to SDR conversion",
        graph_title_size=None,
        xlabel="Input Luminance [cd/m2]",
        ylabel="Output Luminance [cd/m2]",
        axis_label_size=None,
        legend_size=17,
        xlim=[0.009, 15000],
        ylim=[0.009, 15000],
        xtick=None,
        ytick=None,
        xtick_size=None,
        ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None,
        return_figure=True)
    pu.log_scale_settings(ax1, grid_alpha=0.5, bg_color="#E0E0E0")

    img = read_image("./img/dv17_fig1_sdr_out_st2084.tif")[0, :, 0]
    label = "(a) src: ST2084(.tif)"
    x_luminance = tf.eotf_to_luminance(x, tf.ST2084)
    y_luminance = tf.eotf_to_luminance(img, tf.GAMMA24)
    ax1.plot(x_luminance, y_luminance, color=pu.BLUE, label=label)

    # img = read_image("./img/dv17_fig1_203_sdr_out_st2084.tif")[0, :, 0]
    # label = "(b) src: ST2084(.tif), ref-white: 203nits"
    # x_luminance = tf.eotf_to_luminance(x, tf.ST2084)
    # y_luminance = tf.eotf_to_luminance(img, tf.GAMMA24)
    # ax1.plot(x_luminance, y_luminance, label=label)

    img = read_image("./img/dv17_fig1_sdr_out_linear.tif")[0, :, 0]
    label = "(b) src: Linear(.exr), This is the expected result."
    y_luminance = tf.eotf_to_luminance(img, tf.GAMMA24)
    x = np.linspace(0, 1, 1920)
    x_luminance = tpg.shaper_func_log2_to_linear(
        x, min_exposure=-12, max_exposure=12)
    ax1.plot(
        x_luminance * 100, y_luminance, '--', color=pu.RED, label=label)

    # img = read_image("./img/dv17_fig1_203_sdr_out_linear.tif")[0, :, 0]
    # label = "src=Linear(.exr), ref-white=203nits"
    # y_luminance = tf.eotf_to_luminance(img, tf.GAMMA24)
    # x = np.linspace(0, 1, 1920)
    # x_luminance = tpg.shaper_func_log2_to_linear(
    #     x, min_exposure=-12, max_exposure=12)
    # ax1.plot(
    #     x_luminance * 100, y_luminance, label=label)

    plt.legend(loc='upper left')
    fname_full = "./img/fig1.png"
    plt.savefig(fname_full, bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close(fig)


def main_func():
    # create_exr_ramp()
    plot_input_drt()
    # check_100nits_code_value_on_st2084()
    # plot_forum_fig1()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
