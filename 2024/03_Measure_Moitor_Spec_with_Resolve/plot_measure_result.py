# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

# import third-party libraries

# import my libraries
import plot_utility as pu
from ty_display_pro_hl import read_measure_result, calculate_elapsed_seconds
from create_tp_for_measure import create_cv_list
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def plot_peak_60s_data():
    csv_name = "./measure_result/measure_result_peak_60s.csv"
    read_data = read_measure_result(csv_name=csv_name)
    elapsed_time = calculate_elapsed_seconds(file_path=csv_name)
    luminance = read_data[..., 3]
    fname = "./img/peak_60s.png"

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=None,
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
    ax1.plot(elapsed_time, luminance, '-o', label="3% Window")

    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc='lower right', save_fname=fname)


def plot_increment_result(condition: str):
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


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # plot_peak_60s_data()

    condition_list = [
        "Desktop", "Movie_HDR", "Game_HDR", "Custom_Color_HDR",
        "DisplayHDR_True_Black", "HDR_Peak_1000"]
    # condition_list = [
    #     "HDR_Peak_1000"]

    for condition in condition_list:
        plot_increment_result(condition=condition)
