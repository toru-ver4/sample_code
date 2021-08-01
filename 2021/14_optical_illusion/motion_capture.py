# -*- coding: utf-8 -*-
"""

```
"""

# import standard libraries
import os
from pathlib import Path

# import third-party libraries
import numpy as np
from colour.utilities import tstack

# import my libraries
import test_pattern_generator2 as tpg
import plot_utility as pu


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

src_dir = "/work/overuse/2021/14_optical_illusion/motion_src/"
i_asset_dir = "/work/overuse/2021/14_optical_illusion/intermediate/"


def create_intermediate_file_name(src_fname):
    stem = Path(src_fname).stem
    return i_asset_dir + str(stem) + ".png"


def create_intermediate_data(fname):
    threshold = 0.4
    img = tpg.img_read_as_float(fname)
    img[img > threshold] = 1.0
    img[img <= threshold] = 0.0

    gray_img = img[..., 1]
    gray_img = np.dstack([gray_img, gray_img, gray_img])
    stem = Path(fname).stem

    dst_fname = i_asset_dir + str(stem) + ".png"

    tpg.img_wirte_float_as_16bit_int(dst_fname, gray_img)


def sum_value_horizontal(img):
    """
    >>> img = np.array([
    ...     [1, 2, 3, 4, 5, 6],
    ...     [1, 2, 3, 4, 5, 6],
    ...     [1, 2, 3, 4, 5, 6],
    ... ])
    >>> sum_value_horizontal(img)
    [21 21 21]
    """
    return np.sum(img, axis=-1)


def sum_value_vertical(img):
    """
    >>> img = np.array([
    ...     [1, 2, 3, 4, 5, 6],
    ...     [1, 2, 3, 4, 5, 6],
    ...     [1, 2, 3, 4, 5, 6],
    ... ])
    >>> sum_value_vertical(img)
    [ 3  6  9 12 15 18]
    """
    return np.sum(img, axis=-2)


def create_pos_list(frame_num):
    h_pos_list = []
    v_pos_list = []
    for idx in range(frame_num):
        st_num = 216000
        src_fname = src_dir + f"sd_src{st_num+idx:08d}.tif"
        fname = create_intermediate_file_name(src_fname)
        img = tpg.img_read_as_float(fname)
        img = img[..., 1]
        img = 1 - img
        h_sum_list = sum_value_horizontal(img)
        v_sum_list = sum_value_vertical(img)
        h_max_idx = np.argmax(v_sum_list)
        v_max_idx = np.argmax(h_sum_list)
        # print(h_max_idx, v_max_idx)
        h_pos_list.append(h_max_idx)
        v_pos_list.append(v_max_idx)

    h_pos_list = np.array(h_pos_list)
    v_pos_list = np.array(v_pos_list)
    pos_list = tstack([h_pos_list, v_pos_list])

    return pos_list


def create_horizontal_movement(
        fps=60, cycle_num=5, cycle_sec=0.6, cycle_sec_offset=0,
        amp=300, amp_offset=550):

    sec = cycle_num * cycle_sec
    x = np.arange(int(round(sec*fps)))
    cycle_frame = int(round(fps * cycle_sec))
    offset_frame = int(round(fps * cycle_sec_offset))
    y = np.sin(2*np.pi*((x-offset_frame)/cycle_frame))

    win_x = np.arange(int(round(cycle_sec*fps)))
    len_win = len(win_x)
    win_y = np.sin(
        2*np.pi*((win_x-offset_frame)/cycle_frame)-np.pi/2)
    for data in win_y:
        print(data)
    win_y = (win_y + 1) / 2

    y[:len_win//2] = y[:len_win//2] * win_y[:len_win//2]
    y[-len_win//2:] = y[-len_win//2:] * win_y[len_win//2:]
    y = y * amp + amp_offset

    return x, y


def plot_xy_data(pos_list):
    h_data = pos_list[..., 0]
    v_data = pos_list[..., 1]
    x = np.arange(len(h_data))
    h_idx_max = 150

    fps = 60
    cycle_sec = 0.6
    cycle_num = 5

    x2, y2 = create_horizontal_movement(
        fps=fps, cycle_num=cycle_num, cycle_sec=cycle_sec,
        cycle_sec_offset=0, amp=300, amp_offset=550)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Horizontal movement",
        graph_title_size=None,
        xlabel="Frame index", ylabel="Y axis",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=5,
        minor_ytick_num=None)
    ax1.plot(x[:h_idx_max], h_data[:h_idx_max], label="Horizontal")
    ax1.plot(x2, y2, label="emulation")
    fname = "./img/horizontal_move.png"
    pu.show_and_save(fig=fig, legend_loc='upper left', save_fname=fname)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Vertical movement",
        graph_title_size=None,
        xlabel="Frame index", ylabel="Y axis",
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
    ax1.plot(x, v_data, label="Vertical")
    fname = "./img/vertical_move.png"
    pu.show_and_save(fig=fig, legend_loc='upper left', save_fname=fname)


def debug():
    frame_num = 360
    # for idx in range(frame_num):
    #     st_num = 216000
    #     fname = src_dir + f"sd_src{st_num+idx:08d}.tif"
    #     print(fname)
    #     create_intermediate_data(fname)

    # pos_list = create_pos_list(frame_num)
    pos_list_fname = "./lut/pos_list.npy"
    # np.save(pos_list_fname, pos_list)

    pos_list = np.load(pos_list_fname)
    # print(pos_list)
    plot_xy_data(pos_list)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug()
