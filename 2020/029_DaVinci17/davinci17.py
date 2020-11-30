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
    #     ['./img/old/test_out_sdr100.tif', 'SDR 100'],
    #     ['./img/old/test_out_hdr500.tif', 'HDR 500'],
    #     ['./img/old/test_out_hdr1000.tif', 'HDR 1000'],
    #     ['./img/old/test_out_hdr2000.tif', 'HDR 2000'],
    #     ['./img/old/test_out_hdr4000.tif', 'HDR 4000'],
    #     ['./img/old/test_out_off.tif', 'DRT OFF']
    # ]
    # check_input_drt_test(
    #     file_list=file_list, graph_name="Input_DRT_Characteristics_w_SDR")

    # file_list = [
    #     ['./img/old/test_out_hdr500.tif', 'HDR 500'],
    #     ['./img/old/test_out_hdr1000.tif', 'HDR 1000'],
    #     ['./img/old/test_out_hdr2000.tif', 'HDR 2000'],
    #     ['./img/old/test_out_hdr4000.tif', 'HDR 4000'],
    #     ['./img/old/test_out_off.tif', 'DRT OFF']
    # ]
    # check_input_drt_test(
    #     file_list=file_list, graph_name="Input_DRT_Characteristics_wo_SDR")

    # file_list = [
    #     ['./img/old/test_out_sdr_er_100-200.tif', 'SDR ER 100/200'],
    #     ['./img/old/test_out_hdr_er_1000-2000.tif', 'HDR ER 1000/2000'],
    #     ['./img/old/test_out_hdr_er_1000-4000.tif', 'HDR ER 1000/4000'],
    #     ['./img/old/test_out_hdr_er_1000-10000.tif', 'HDR ER 1000/10000'],
    #     ['./img/old/test_out_hdr_er_4000-10000.tif', 'HDR ER 4000/10000'],
    #     ['./img/old/test_out_off.tif', 'DRT OFF']
    # ]
    # check_input_drt_test(
    #     file_list=file_list, graph_name="Input_DRT_Characteristics_ER_w_SDR")

    file_list = [
        ['./img/old/test_out_hdr_er_1000-2000.tif', 'HDR ER 1000/2000', '-.'],
        ['./img/old/test_out_hdr_er_1000-4000.tif', 'HDR ER 1000/4000', '--'],
        ['./img/old/test_out_hdr_er_1000-10000.tif', 'HDR ER 1000/10000', '-'],
        ['./img/old/test_out_hdr_er_4000-10000.tif', 'HDR ER 4000/10000', '-'],
        # ['./img/old/test_out_off.tif', 'DRT OFF']
    ]
    check_input_drt_test(
        file_list=file_list, graph_name="Input_DRT_Characteristics_ER_wo_SDR")

    # check_input_drt_test_sdr_only()


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
        ls = file_list[idx][2]
        y_luminance = tf.eotf_to_luminance(img, tf.ST2084)
        ax1.plot(x_luminance, y_luminance, ls, label=label)

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

    # img = read_image("./img/test_out_sdr100_on_gm24.tif")[0, :, 0]
    # label = "DRT OFF(ST2084 to Gamma2.4 (.tif))"
    # x_luminance = tf.eotf_to_luminance(x, tf.ST2084)
    # y_luminance = tf.eotf_to_luminance(img, tf.GAMMA24)
    # ax1.plot(x_luminance, y_luminance, label=label)

    # img = read_image("./img/test_out_sdr100_on_gm24_203nits.tif")[0, :, 0]
    # label = "DRT OFF(ST2084 to Gamma2.4 (.tif) 203nits)"
    # x_luminance = tf.eotf_to_luminance(x, tf.ST2084)
    # y_luminance = tf.eotf_to_luminance(img, tf.GAMMA24)
    # ax1.plot(x_luminance, y_luminance, label=label)

    img = read_image("./img/old/test_out_sdr100_on_gm24.tif")[0, :, 0]
    label = 'SDR 100 (Output color space is Gamma2.4)'
    x_luminance = tf.eotf_to_luminance(x, tf.ST2084)
    y_luminance = tf.eotf_to_luminance(img, tf.GAMMA24)
    ax1.plot(x_luminance, y_luminance, label=label)

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


def plot_output_drt():
    # file_list = [
    #     # ['./img/Output_DRT_SDR_ER_100-200.tif', 'SDR ER 100/200', '-'],
    #     ['./img/old/Output_DRT_HDR_ER_1000-2000.tif', 'HDR ER 1000/2000', '-'],
    #     ['./img/old/Output_DRT_HDR_ER_1000-4000.tif', 'HDR ER 1000/4000', '-'],
    #     ['./img/old/Output_DRT_HDR_ER_1000-10000.tif', 'HDR ER 1000/10000', '-'],
    #     ['./img/old/Output_DRT_HDR_ER_4000-10000.tif', 'HDR ER 4000/10000', '--'],
    # ]
    # check_output_drt_test(
    #     file_list=file_list,
    #     graph_name="DaVinci17 Output DRT ER 無印ST2084")

    # file_list = [
    #     # ['./img/Output_DRT_SDR_ER_100-200.tif', 'SDR ER 100/200', '-'],
    #     ['./img/Output_DRT_HDR_ER_1000-2000.tif', 'HDR ER 1000/2000', '-'],
    #     ['./img/Output_DRT_HDR_ER_1000-4000.tif', 'HDR ER 1000/4000', '-'],
    #     ['./img/Output_DRT_HDR_ER_1000-10000.tif', 'HDR ER 1000/10000', '-'],
    #     ['./img/Output_DRT_HDR_ER_4000-10000.tif', 'HDR ER 4000/10000', '--'],
    # ]
    # check_output_drt_test(
    #     file_list=file_list,
    #     graph_name="DaVinci17 Output DRT Characteristics ER")

    # file_list = [
    #     # ['./img/Output_DRT_SDR_100.tif', 'SDR 100', '-'],
    #     ['./img/old/Output_DRT_HDR_500.tif', 'HDR 500', '-'],
    #     ['./img/old/Output_DRT_HDR_1000.tif', 'HDR 1000', '-'],
    #     ['./img/old/Output_DRT_HDR_2000.tif', 'HDR 2000', '-'],
    #     ['./img/old/Output_DRT_HDR_4000.tif', 'HDR 4000', '-']
    # ]
    # check_output_drt_test(
    #     file_list=file_list,
    #     graph_name="DaVinci17 Output DRT 無印 ST2084")

    file_list = [
        # ['./img/Output_DRT_SDR_100.tif', 'SDR 100', '-'],
        ['./img/Output_DRT_HDR_500.tif', 'HDR 500', '-'],
        ['./img/Output_DRT_HDR_1000.tif', 'HDR 1000', '-'],
        ['./img/Output_DRT_HDR_2000.tif', 'HDR 2000', '-'],
        ['./img/Output_DRT_HDR_4000.tif', 'HDR 4000', '-'],
        ['./img/Output_DRT_HDR_10000.tif', 'Custom (10000 nit)', '--']
    ]
    check_output_drt_test(
        file_list=file_list,
        graph_name="DaVinci17 Output DRT Characteristics")

    file_list = [
        ['./img/DRT_In_None_HDR1000-500.tif', 'HDR 1000, ST2084 500 nit', '-'],
        ['./img/DRT_In_None_HDR1000-1000.tif', 'HDR 1000, ST2084 1000 nit', '-'],
        ['./img/DRT_In_None_HDR1000-2000.tif', 'HDR 1000, ST2084 2000 nit', '-'],
        ['./img/DRT_In_None_HDR1000-4000.tif', 'HDR 1000, ST2084 4000 nit', '-'],
        ['./img/DRT_In_None_HDR1000-10000.tif', 'HDR 1000, ST2084 10000 nit', '-'],
    ]
    check_output_drt_test(
        file_list=file_list,
        graph_name="DaVinci17 Out DRT Characteristics_fix_HDR1000")


def check_output_drt_test(file_list, graph_name):
    x = np.linspace(0, 1, 1920)
    x_luminance = tf.eotf_to_luminance(x, tf.ST2084)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="DaVinci17 Output DRT Characteristics",
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

    for idx in range(len(file_list)):
        img = read_image(file_list[idx][0])[0, :, 0]
        label = file_list[idx][1]
        ls = file_list[idx][2]
        y_luminance = tf.eotf_to_luminance(img, tf.ST2084)
        ax1.plot(x_luminance, y_luminance, ls, label=label)

    plt.legend(loc='upper left')
    fname_full = f"./img/{graph_name}.png".replace(' ', "_")
    plt.savefig(fname_full, bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close(fig)


def check_output_drt_test_exr(file_list, graph_name):
    x = np.linspace(0, 1, 1920)
    x_luminance = tf.eotf_to_luminance(x, tf.ST2084)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title=graph_name,
        graph_title_size=None,
        xlabel="Input Luminance [cd/m2]",
        ylabel="Output Luminance [cd/m2]",
        axis_label_size=None,
        legend_size=17,
        xlim=[0.009, 15000],
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None,
        ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None,
        return_figure=True)
    pu.log_scale_settings(ax1, grid_alpha=0.5, bg_color="#E0E0E0")

    for idx in range(len(file_list)):
        img = read_image(file_list[idx][0])[0, :, 0]
        label = file_list[idx][1]
        ls = file_list[idx][2]
        y_luminance = img * 10000
        ax1.plot(x_luminance, y_luminance, ls, label=label)

    plt.legend(loc='upper left')
    fname_full = f"./img/{graph_name}.png".replace(' ', "_")
    plt.savefig(fname_full, bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close(fig)


def plot_total_drt():
    file_list = [
        ['./img/DRT_Total_HDR_500.tif', 'HDR 500', '-'],
        ['./img/DRT_Total_HDR_1000.tif', 'HDR 1000', '-'],
        ['./img/DRT_Total_HDR_2000.tif', 'HDR 2000', '-'],
        ['./img/DRT_Total_HDR_4000.tif', 'HDR 4000', '-'],
        ['./img/DRT_Total_HDR_10000.tif', 'Custom (10000 nit)', '-'],
    ]
    check_total_drt_test(
        file_list=file_list,
        graph_name="Input-Output_DRT_Characteristics")

    file_list = [
        ['./img/Output_DRT_HDR1000-500.tif',  'HDR 1000, ST2084 500 nit', '-'],
        ['./img/Output_DRT_HDR1000-1000.tif', 'HDR 1000, ST2084 1000 nit', '-'],
        ['./img/Output_DRT_HDR1000-2000.tif', 'HDR 1000, ST2084 2000 nit', '-'],
        ['./img/Output_DRT_HDR1000-4000.tif', 'HDR 1000, ST2084 4000 nit', '-'],
        ['./img/Output_DRT_HDR1000-10000.tif','HDR 1000, ST2084 10000 nit', '-'],
    ]
    check_total_drt_test(
        file_list=file_list,
        graph_name="DaVinci17 In-Out DRT Characteristics_fix_HDR1000")

    file_list = [
        ['./img/DRT_Total_HDR_ER_1000-2000.tif', 'HDR ER 1000/2000', '-'],
        ['./img/DRT_Total_HDR_ER_1000-4000.tif', 'HDR ER 1000/4000', '-'],
        ['./img/DRT_Total_HDR_ER_1000-10000.tif', 'HDR ER 1000/10000', '-'],
        ['./img/DRT_Total_HDR_ER_4000-10000.tif', 'HDR ER 4000/10000', '-'],
    ]
    check_total_drt_test(
        file_list=file_list,
        graph_name="Input-Output_DRT_Characteristics_ER")


def check_total_drt_test(file_list, graph_name):
    x = np.linspace(0, 1, 1920)
    x_luminance = tf.eotf_to_luminance(x, tf.ST2084)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="DaVinci17 Input-Output DRT Characteristics",
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

    for idx in range(len(file_list)):
        img = read_image(file_list[idx][0])[0, :, 0]
        label = file_list[idx][1]
        ls = file_list[idx][2]
        y_luminance = tf.eotf_to_luminance(img, tf.ST2084)
        ax1.plot(x_luminance, y_luminance, ls, label=label)

    plt.legend(loc='upper left')
    fname_full = f"./img/{graph_name}.png".replace(' ', "_")
    plt.savefig(fname_full, bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close(fig)


def plot_inv_drt():
    file_list = [
        # ['./img/Inverse_DRT_to_HDR500.tif', 'SDR to HDR 500 nit', '-'],
        ['./img/Inverse_DRT_to_HDR1000.tif', 'SDR to HDR 1000 nit', '-'],
        # ['./img/Inverse_DRT_to_HDR2000.tif', 'SDR to HDR 2000 nit', '-'],
        ['./img/Inverse_DRT_to_HDR4000.tif', 'SDR to HDR 4000 nit', '-'],
        ['./img/Inverse_DRT_to_HDR10000.tif', 'SDR to HDR 10000 nit', '-'],
    ]
    check_inv_drt_test(
        file_list=file_list,
        graph_name="Inverse_DRT_Characteristics")


def check_inv_drt_test(file_list, graph_name):
    x = np.linspace(0, 1, 1920)
    x_luminance = tf.eotf_to_luminance(x, tf.GAMMA24)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="DaVinci17 Inverse DRT for SDR to HDR Conversion",
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
        ls = file_list[idx][2]
        y_luminance = tf.eotf_to_luminance(img, tf.ST2084)
        ax1.plot(x_luminance, y_luminance, ls, label=label)

    plt.legend(loc='upper left')
    fname_full = f"./img/{graph_name}.png".replace(' ', "_")
    plt.savefig(fname_full, bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close(fig)


def main_func():
    # create_exr_ramp()
    plot_input_drt()
    # plot_output_drt()
    # check_100nits_code_value_on_st2084()
    # plot_forum_fig1()
    # plot_total_drt()
    # plot_inv_drt()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
