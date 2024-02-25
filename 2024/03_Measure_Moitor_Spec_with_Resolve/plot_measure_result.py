# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
import re

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# import my libraries
import plot_utility as pu
from ty_utility import search_specific_extension_files
from ty_display_pro_hl import read_measure_result, calculate_elapsed_seconds
from create_tp_for_measure import create_cv_list
import transfer_functions as tf
import test_pattern_generator2 as tpg
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def plot_peak_60s_data():
    window_size_list = [3, 5, 10, 50, 100]
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Short-duration Test",
        graph_title_size=None,
        xlabel="Elapsed Time [sec]",
        ylabel="Luminance [nits]",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=[0, 1050],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,    
        minor_xtick_num=None,
        minor_ytick_num=None)
    for window_size in window_size_list:
        csv_name = f"./AW3225QF/measure_duration_{window_size:03d}.csv"
        read_data = read_measure_result(csv_name=csv_name)
        elapsed_time = calculate_elapsed_seconds(file_path=csv_name)
        luminance = read_data[..., 3]
        ax1.plot(
            elapsed_time, luminance, '-o', label=f"{window_size}% Window")

    fname = "./img/duration_test_result.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc='lower right', save_fname=fname, ncol=2,
        show=True)


def plot_each_hdr_mode_result(condition: str):
    percent_list = [1.00, 0.50, 0.20, 0.10, 0.03]
    color_list = [pu.GRAY50, pu.BLUE, pu.GREEN, pu.RED, pu.MAJENTA]
    fname = f"./img/increment_patch_{condition}.png"
    cv_list = create_cv_list(num_of_block=64)
    ref_lumnance = tf.eotf_to_luminance(cv_list/1023, tf.ST2084)
    condition_str_space = condition.replace("_", " ")

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=f"Smart HDR Settings: {condition_str_space}",
        graph_title_size=None,
        xlabel="Target Luminance [nits]",
        ylabel="Measured Luminance [nits]",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=2,
        minor_xtick_num=None,
        minor_ytick_num=None)
    for idx, percent in enumerate(percent_list):
        percent_str = f"{int(percent * 100):03d}"
        csv_name = f"./AW3225QF/measure_{condition}_{percent_str}_patch.csv"
        read_data = read_measure_result(csv_name=csv_name)
        luminance = read_data[..., 3]
        label = f"{int(percent*100)}% Window"
        ax1.plot(
            ref_lumnance, luminance, '-o', color=color_list[idx], label=label)
    ax1.plot(ref_lumnance, ref_lumnance, '--k', lw=1.5, label="Reference")
    pu.log_scale_settings(ax1=ax1, grid_alpha=0.5, bg_color="#F0F0F0")
    ax1.set_xlim(0.008, 11000)
    ax1.set_ylim(0.008, 11000)

    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', save_fname=fname, show=True)
    

def scatter_plot_for_single_patch(
        data, patch_idx, ref_xy, patch_color,
        luminance_list, window_size_list):
    diff_data = np.zeros_like(data)
    diff_ref = np.zeros_like(ref_xy)
    idx_1k_3p_win = 20
    print(data[..., 0])
    print(data[..., 1])
    # diff_data[..., 0] = ref_xy[0] - data[..., 0]
    # diff_data[..., 1] = ref_xy[1] - data[..., 1]
    diff_data[..., 0] = data[..., 0] - data[idx_1k_3p_win, 0]
    diff_data[..., 1] = data[..., 1] - data[idx_1k_3p_win, 1]
    diff_ref[0] = ref_xy[..., 0] - data[idx_1k_3p_win, 0]
    diff_ref[1] = ref_xy[..., 1] - data[idx_1k_3p_win, 1]
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=f"Single Patch Scatter Plot - {patch_idx:02d}",
        graph_title_size=None,
        xlabel="x (Difference from 3%, 1000 nits Patch)",
        ylabel="y (Difference from 3%, 1000 nits Patch)",
        axis_label_size=None,
        legend_size=17,
        xlim=[-0.015, 0.015],
        ylim=[-0.015, 0.015],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=2,
        minor_xtick_num=None,
        minor_ytick_num=None)
    lumi_rate = [0.2, 0.4, 0.6, 0.8, 1.0]
    size_list = np.array([14, 19, 23, 26, 29]) * 1.6
    # size_list = [28, 38, 46, 52, 58]
    base_cnt = 0
    for l_idx, luminance in enumerate(luminance_list):
        # for w_idx, window_size in enumerate(window_size_list):
        for w_idx in range(4, -1, -1):
            cnt = base_cnt + w_idx
            window_size = window_size_list[w_idx]
            label = f"{luminance}-nits, {window_size}% window"
            plot_color = np.array(patch_color) * lumi_rate[l_idx]
            edge_color = (0.7, 0.7, 0.7) if np.max(plot_color) < 0.4 else 'k'
            ax1.plot(
                diff_data[cnt, 0], diff_data[cnt, 1], 's', label=None,
                ms=size_list[w_idx], mec=edge_color, mfc=plot_color, mew=1.6)
            # print(f"cnt = {cnt}, w_idx = {w_idx}, w_size = {size_list[w_idx]}")
        base_cnt += 5

    ax1.plot(
        diff_ref[0], diff_ref[1], "x", label="Reference",
        ms=30, mec=patch_color, mew=6)

    ax1.minorticks_on()
    ax1.grid(which='minor', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.001))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.001))

    fname = f"./img/scatter_single_patch_{patch_idx:02d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', save_fname=fname, show=False,
        fontsize=20)
    

def create_cc_patch_measure_result_fname(luminance, window_size):
    window_size_int = int(window_size * 100)
    fname = f"./AW3225QF/cc_measure_lumi-{luminance:04d}_"
    fname += f"win-{window_size_int:03d}.csv"

    return fname


def create_apl_cc_patch_measure_result_fname(luminance, window_size):
    window_size_int = int(window_size * 100)
    fname = f"./AW3225QF/apl_cc_measure_lumi-{luminance:04d}_"
    fname += f"win-{window_size_int:03d}.csv"

    return fname


def plot_color_checker_multi_size_and_cv():
    luminance_list = [100, 200, 400, 600, 1000]
    window_size_list = [0.03, 0.10, 0.20, 0.50, 1.00]
    num_of_cc_patch = 18

    cc_xyY = tpg.generate_color_checker_xyY_value()
    cc_rgb = tpg.generate_color_checker_rgb_value()
    cc_rgb = tf.oetf(np.clip(cc_rgb, 0, 1), tf.SRGB)
    for idx in range(num_of_cc_patch):
        buf = []
        for luminance in luminance_list:
            for window_size in window_size_list:
                csv_name = create_cc_patch_measure_result_fname(
                    luminance=luminance, window_size=window_size)
                data = read_measure_result(csv_name=csv_name)
                # add xyY
                temp_buf = [data[idx, 4], data[idx, 5], data[idx, 3]]
                buf.append(temp_buf)
        measured_xyY = np.array(buf)
        # for y_idx, xyY in enumerate(measured_xyY):
        #     print(f"{y_idx}, {xyY}")
        scatter_plot_for_single_patch(
            data=measured_xyY, patch_idx=idx, ref_xy=cc_xyY[idx, :2],
            patch_color=cc_rgb[idx],
            luminance_list=luminance_list, window_size_list=window_size_list)
        # break


def plot_color_checker_with_over_apl():
    luminance = 1000
    window_size_list = [1.00, 0.50, 0.20, 0.10, 0.03]
    num_of_cc_patch = 18

    # cc_xyY = tpg.generate_color_checker_xyY_value()
    cc_rgb = tpg.generate_color_checker_rgb_value()
    cc_rgb = tf.oetf(np.clip(cc_rgb, 0, 1), tf.SRGB)
    for cc_idx in range(num_of_cc_patch):
        buf = []
        for window_size in window_size_list:
            csv_name = create_cc_patch_measure_result_fname(
                luminance=luminance, window_size=window_size)
            data = read_measure_result(csv_name=csv_name)
            # add xyY
            temp_buf = [data[cc_idx, 4], data[cc_idx, 5], data[cc_idx, 3]]
            buf.append(temp_buf)
        measured_xyY = np.array(buf)
        # for y_idx, xyY in enumerate(measured_xyY):
        #     print(f"{y_idx}, {xyY}")
        plot_color_checker_with_over_apl_core(
            cc_idx=cc_idx, data=measured_xyY, patch_color=cc_rgb[cc_idx],
            window_size_list=window_size_list)
        plot_color_checker_Y_with_over_apl_core(
            cc_idx=cc_idx, data=measured_xyY, patch_color=cc_rgb[cc_idx],
            window_size_list=window_size_list)
        # break
        
    concat_apl_cc_plot_data()
    concat_apl_cc_Y_plot_data()
        

def concat_apl_cc_plot_data():
    v_buf = []
    for v_idx in range(3):
        h_buf = []
        for h_idx in range(6):
            idx = v_idx * 6 + h_idx
            fname = f"./img/APL_Patch_{idx+1:02d}.png"
            img = tpg.img_read_as_float(fname)
            h_buf.append(img)
        v_buf.append(np.hstack(h_buf))
    out_img = np.vstack(v_buf)

    tpg.img_wirte_float_as_16bit_int(
        "./img/APL_Patch_all.png", out_img)


def plot_color_checker_with_over_apl_core(
        cc_idx, data, patch_color, window_size_list):
    diff_data = np.zeros_like(data)
    diff_data[..., 0] = data[..., 0] - data[-1, 0]
    diff_data[..., 1] = data[..., 1] - data[-1, 1]
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(9, 9),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=f"APL and Color Difference - {cc_idx+1:02d}",
        graph_title_size=None,
        xlabel="x (Difference from 3% Patch)",
        ylabel="y (Difference from 3% Patch)",
        axis_label_size=None,
        legend_size=17,
        xlim=[-0.003, 0.003],
        ylim=[-0.003, 0.003],
        xtick_size=None, ytick_size=None,
        linewidth=2,
        minor_xtick_num=None,
        minor_ytick_num=None)
    
    size_list = np.array([29, 26, 23, 19, 14]) * 1.5
    alpha_list = [0.3, 0.5, 0.7, 0.8, 1.0]
    marker_list = ['x', 'x', "x", 'x', "x"]
    color_rate = np.linspace(0, 1, 5)
    ms_list = np.array([20, 20, 20, 20, 20]) * 1.4
    for w_idx, window_size in enumerate(window_size_list):
        label = f"{int(window_size*100)}% window"
        plot_color = np.array(patch_color) * color_rate[w_idx]
        edge_color = (0.7, 0.7, 0.7) if np.max(plot_color) < 0.6 else 'k'
        ax1.plot(
            diff_data[w_idx, 0], diff_data[w_idx, 1], marker_list[w_idx],
            label=label, mec=plot_color, mfc=plot_color, mew=4,
            ms=ms_list[w_idx],
            # ms=size_list[w_idx], mec=edge_color, mfc=plot_color, mew=1.6,
            # alpha=alpha_list[w_idx]
        )
        
    fname = f"./img/APL_Patch_{cc_idx+1:02d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', save_fname=fname, show=False,
        fontsize=16)


def plot_color_checker_Y_with_over_apl_core(
        cc_idx, data, patch_color, window_size_list):
    num_of_data = len(data)
    categories = [f"{int(size * 100)}%" for size in window_size_list]

    x = np.linspace(0, 1, num_of_data)[::-1]
    diff_data = np.zeros_like(data)
    diff_data[..., 2] = data[..., 2]
    y = diff_data[..., 2]
    y = y[::-1]

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(9, 9),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=f"Luminance (Color Patch - {cc_idx+1:02d})",
        graph_title_size=None,
        xlabel="Window Size",
        ylabel="Luminance [nits]",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=[0, 650],
        xtick_size=None, ytick_size=None,
        linewidth=2,
        minor_xtick_num=None,
        minor_ytick_num=None)
    
    plot_color = np.array(patch_color)
    ax1.bar(
        categories, y,
        width=0.5, color=plot_color, edgecolor='k')
        
    fname = f"./img/APL_Patch_Y_{cc_idx+1:02d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, save_fname=fname, show=False,
        fontsize=20)


def concat_cc_plot_data():
    v_buf = []
    for v_idx in range(3):
        h_buf = []
        for h_idx in range(6):
            idx = v_idx * 6 + h_idx
            fname = f"./img/scatter_single_patch_{idx+1:02d}.png"
            img = tpg.img_read_as_float(fname)
            h_buf.append(img)
        v_buf.append(np.hstack(h_buf))
    out_img = np.vstack(v_buf)

    tpg.img_wirte_float_as_16bit_int(
        "./img/scatter_single_patch_all.png", out_img)


def concat_apl_cc_Y_plot_data():
    v_buf = []
    for v_idx in range(3):
        h_buf = []
        for h_idx in range(6):
            idx = v_idx * 6 + h_idx
            fname = f"./img/APL_Patch_Y_{idx+1:02d}.png"
            img = tpg.img_read_as_float(fname)
            h_buf.append(img)
        v_buf.append(np.hstack(h_buf))
    out_img = np.vstack(v_buf)

    tpg.img_wirte_float_as_16bit_int(
        "./img/APL_Patch_Y_all.png", out_img)


def check_ccss_difference_with_white_patch():
    measure_file_name = "./AW3225QF/white_ccss_diff.csv"
    ccss_file_list = search_specific_extension_files(
        dir="./ccss", ext=".ccss")
    marker_size = 12
    pattern = r"ccss\\|_[0-9]+[a-zA-Z]+[0-9]+\.ccss"
    print(ccss_file_list)

    data_list = read_measure_result(csv_name=measure_file_name)

    x_min = np.min(data_list[..., 4])
    x_max = np.max(data_list[..., 4])
    y_min = np.min(data_list[..., 5])
    y_max = np.max(data_list[..., 5])

    rad = np.linspace(0, 1, 1024) * 2 * np.pi
    x_01 = np.cos(rad) * 0.001 + cs.D65[0]
    x_02 = np.cos(rad) * 0.002 + cs.D65[0]
    y_01 = np.sin(rad) * 0.001 + cs.D65[1]
    y_02 = np.sin(rad) * 0.002 + cs.D65[1]

    x_diff = max(abs(cs.D65[0]-x_min), abs(cs.D65[0]-x_max)) * 1.2
    y_diff = max(abs(cs.D65[1]-y_min), abs(cs.D65[1]-y_max)) * 1.2
    diff_d65 = max(x_diff, y_diff)

    print(x_diff, y_diff)

    title = "The Relationship Between the CCSS File "
    title += "and the D65 Measurement Values"
    fig, ax1 = pu.plot_1_graph(
        fontsize=12,
        figsize=(9, 9),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=title,
        graph_title_size=None,
        xlabel="x",
        ylabel="y",
        axis_label_size=None,
        legend_size=17,
        xlim=[cs.D65[0]-diff_d65, cs.D65[0]+diff_d65],
        ylim=[cs.D65[1]-diff_d65, cs.D65[1]+diff_d65],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    for data, ccss_file in zip(data_list, ccss_file_list):
        x = data[..., 4]
        y = data[..., 5]
        label = re.sub(pattern, '', ccss_file)
        ax1.plot(x, y, 'o', ms=marker_size, label=label)
    ax1.plot(
        cs.D65[0], cs.D65[1], 'x', color='k', label="D65",
        ms=marker_size, mew=4)
    ax1.plot(x_01, y_01, '--k', label="Δ0.001", lw=2, alpha=0.66)
    ax1.plot(x_02, y_02, '-.k', label="Δ0.002", lw=2, alpha=0.66)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', fontsize=12,
        save_fname="./img/d65_with_all_ccss.png", show=True)
    

def plot_abs_diff_colorchecker():
    num_of_color_checker = 18
    cc_xyY = tpg.generate_color_checker_xyY_value()
    cc_rgb = tpg.generate_color_checker_rgb_value()
    cc_rgb = tf.oetf(np.clip(cc_rgb, 0, 1), tf.SRGB)
    luminance_list = [100, 200]
    window_size = 0.03
    for cc_idx in range(num_of_color_checker):
        buf = []
        for luminance in luminance_list:
            csv_name = create_cc_patch_measure_result_fname(
                luminance=luminance, window_size=window_size)
            data = read_measure_result(csv_name=csv_name)
            # add xyY
            temp_buf = [data[cc_idx, 4], data[cc_idx, 5], data[cc_idx, 3]]
            buf.append(temp_buf)
        measured_xyY = np.array(buf)
        plot_abs_diff_colorchecker_core(
            cc_idx=cc_idx, data=measured_xyY, ref_xy=cc_xyY[cc_idx],
            patch_color=cc_rgb[cc_idx])
        # break

    concat_abs_diff_colorchecker()


def concat_abs_diff_colorchecker():
    v_buf = []
    for v_idx in range(3):
        h_buf = []
        for h_idx in range(6):
            idx = v_idx * 6 + h_idx
            fname = f"./img/abs_xy_cc_patch_{idx+1:02d}.png"
            img = tpg.img_read_as_float(fname)
            h_buf.append(img)
        v_buf.append(np.hstack(h_buf))
    out_img = np.vstack(v_buf)

    tpg.img_wirte_float_as_16bit_int(
        "./img/abs_xy_cc_patch_all.png", out_img)


def plot_abs_diff_colorchecker_core(cc_idx, data, ref_xy, patch_color):
    max_diff = 0.025
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(9, 9),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=f"Color Accuracy for the ColorChecker Patch - {cc_idx+1:02d}",
        graph_title_size=None,
        xlabel="x",
        ylabel="y",
        axis_label_size=None,
        legend_size=17,
        xlim=[ref_xy[0] - max_diff, ref_xy[0] + max_diff],
        ylim=[ref_xy[1] - max_diff, ref_xy[1] + max_diff],
        xtick_size=None, ytick_size=None,
        linewidth=2,
        minor_xtick_num=None,
        minor_ytick_num=None)
    
    ax1.plot(
        ref_xy[0], ref_xy[1], "s", label="Reference",
        mec=patch_color, mfc='w', mew=4, ms=20,
    )
    ax1.plot(
        data[0, 0], data[0, 1], "x", label="White: 100 nits",
        mec=patch_color*0.5, mfc=patch_color*0.5, mew=4, ms=20,
    )
    ax1.plot(
        data[1, 0], data[1, 1], "+", label="White: 200 nits",
        mec=patch_color, mfc=patch_color, mew=4, ms=24,
    )
        
    fname = f"./img/abs_xy_cc_patch_{cc_idx+1:02d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', save_fname=fname, show=False,
        fontsize=16)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    plot_peak_60s_data()

    # condition_list = [
    #     "Desktop", "Movie_HDR", "Game_HDR", "Custom_Color_HDR",
    #     "DisplayHDR_True_Black", "HDR_Peak_1000"]
    # for condition in condition_list:
    #     plot_each_hdr_mode_result(condition=condition)

    # plot_color_checker_multi_size_and_cv()
    # concat_cc_plot_data()

    # plot_color_checker_with_over_apl()

    # check_ccss_difference_with_white_patch()

    # plot_abs_diff_colorchecker()
