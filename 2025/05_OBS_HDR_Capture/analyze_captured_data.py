# -*- coding: utf-8 -*-

# import standard libraries
import sys
import os
from pathlib import Path

# import third-party libraries
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import my libraries
import test_pattern_generator2 as tpg
import plot_utility as pu


def get_ramp_from_jxr_data():
    fname = "./capture_data/check_precision/Microsoft_Edge.png"
    num_of_sample = 65
    img = np.round(tpg.img_read_as_float(fname) * 1023).astype(np.uint16)

    h_pos_st = 172
    h_pos_ed = 1408
    v_pos_1 = 1026
    v_pos_2 = 1371
    still_h_pos_list = np.linspace(h_pos_st, h_pos_ed, num_of_sample, dtype=np.uint16)
    still_data = img[v_pos_1, still_h_pos_list, 1]
    video_data = img[v_pos_2, still_h_pos_list, 1]

    return still_data, video_data


def get_ramp_from_display_capture():
    fname = "./capture_data/check_precision/Browser-Game_Display_Capture_80-10000.png"
    num_of_sample = 65
    img = np.round(tpg.img_read_as_float(fname) * 1023).astype(np.uint16)

    h_pos_st = 172
    h_pos_ed = 1408
    v_pos_1 = 1026
    v_pos_2 = 1371
    still_h_pos_list = np.linspace(h_pos_st, h_pos_ed, num_of_sample, dtype=np.uint16)
    still_data = img[v_pos_1, still_h_pos_list, 1]
    video_data = img[v_pos_2, still_h_pos_list, 1]

    return still_data, video_data


def get_ramp_from_window_game_capture():
    fname = "./capture_data/check_precision/Browser-Game_Window-Game_Capture_80-10000.png"
    num_of_sample = 65
    img = np.round(tpg.img_read_as_float(fname) * 1023).astype(np.uint16)

    h_pos_st = 174
    h_pos_ed = 1448
    v_pos_1 = 1054
    v_pos_2 = 1414
    still_h_pos_list = np.linspace(h_pos_st, h_pos_ed, num_of_sample, dtype=np.uint16)
    still_data = img[v_pos_1, still_h_pos_list, 1]
    video_data = img[v_pos_2, still_h_pos_list, 1]

    return still_data, video_data


def plot_data():
    ref_still, ref_video = get_ramp_from_jxr_data()
    disp_cap_still, disp_cap_video = get_ramp_from_display_capture()
    win_cap_still, win_cap_video = get_ramp_from_window_game_capture()
    x = [x * 16 for x in range(64)] + [1023]

    xtics = [x * 128 for x in range(8)] + [1023]
    ytics = [x * 128 for x in range(8)] + [1023]

    fig, ax1 = pu.plot_1_graph(
        fontsize=18,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="YouTube Video Capture Result on Edge",
        graph_title_size=None,
        xlabel="Target Code Value (10-bit)",
        ylabel="Captured Code Value (10-bit)",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=xtics,
        ytick=ytics,
        xtick_size=None, ytick_size=None,
        linewidth=1,
        minor_xtick_num=None,
        minor_ytick_num=None
    )
    ax1.plot(x, ref_video, '-o', label="Reference (HDR screenshot on Windows)", color=pu.RED)
    ax1.plot(x, disp_cap_video, '-o', label="OBS Display Capture", color=pu.GREEN)
    ax1.plot(x, win_cap_video, '-o', label="OBS Window Capture", color=pu.BLUE)
    save_fname = "./img/captured_video_result.png"
    pu.show_and_save(
        fig=fig, legend_loc='upper left', save_fname=save_fname)

    fig, ax1 = pu.plot_1_graph(
        fontsize=18,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Still Image Capture Result on Edge",
        graph_title_size=None,
        xlabel="Target Code Value (10-bit)",
        ylabel="Captured Code Value (10-bit)",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=[-50, 1073],
        xtick=xtics,
        ytick=ytics,
        xtick_size=None, ytick_size=None,
        linewidth=1,
        minor_xtick_num=None,
        minor_ytick_num=None
    )
    ax1.plot(x, ref_still, '-o', label="Reference (HDR screenshot on Windows)", color=pu.RED)
    ax1.plot(x, disp_cap_still, '-o', label="OBS Display Capture", color=pu.GREEN)
    ax1.plot(x, win_cap_still, '-o', label="OBS Window Capture", color=pu.BLUE)
    save_fname = "./img/captured_still_result.png"
    pu.show_and_save(
        fig=fig, legend_loc='upper left', save_fname=save_fname)


#####################
# Main
#####################
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    plot_data()
