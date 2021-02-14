# -*- coding: utf-8 -*-
"""
fix alpha blending of font_control
===================================
"""

# import standard libraries
from operator import sub
import os
from pathlib import Path

# import third-party libraries
import numpy as np
from colour import LUT3D, RGB_to_XYZ, XYZ_to_Lab
from colour.models import BT709_COLOURSPACE
from colour.difference import delta_E_CIE2000

# import my libraries
import test_pattern_generator2 as tpg
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


DE2000_8BIT_FILE = "./de2000_8bit.npy"
DE2000_RANK_INDEX = "./de2000_8bit_rank.npy"


def bt709_to_lab(rgb_gm24):
    rgb_linear = rgb_gm24 ** 2.4
    large_xyz = RGB_to_XYZ(
        rgb_linear, cs.D65, cs.D65, BT709_COLOURSPACE.RGB_to_XYZ_matrix)
    lab = XYZ_to_Lab(large_xyz)

    return lab


def get_top_nth_index(data, inner_grid_num, direction_num, nth=10):
    """
    get the top n-th index.
    """
    r_base = (inner_grid_num ** 2) * direction_num
    g_base = inner_grid_num * direction_num
    b_base = direction_num
    arg_idx = np.argsort(data.flatten())[::-1]
    arg_idx = arg_idx[:nth]
    rr_idx = arg_idx // r_base
    gg_idx = (arg_idx % r_base) // g_base
    bb_idx = (arg_idx % g_base) // b_base
    dd_idx = (arg_idx % b_base)

    return (rr_idx, gg_idx, bb_idx, dd_idx)


def calc_delta_e_for_next_grid(grid_num):
    # calc lab
    rgb_gm24 = LUT3D.linear_table(size=grid_num)
    lab = bt709_to_lab(rgb_gm24)

    # XY平面は同時計算して Z方向は forループで処理する
    inner_grid_num = grid_num-1
    inner_idx_list = np.arange(inner_grid_num)
    inner_grid_num = len(inner_idx_list)
    sum_diff_r_direction_buf = []
    for r_idx in inner_idx_list:
        base_lab = lab[r_idx, :inner_grid_num, :inner_grid_num]
        lab_next1 = lab[r_idx+0, 0:inner_grid_num+0, 1:inner_grid_num+1]
        lab_next2 = lab[r_idx+0, 1:inner_grid_num+1, 0:inner_grid_num+0]
        lab_next3 = lab[r_idx+0, 1:inner_grid_num+1, 1:inner_grid_num+1]
        lab_next4 = lab[r_idx+1, 0:inner_grid_num+0, 0:inner_grid_num+0]
        lab_next5 = lab[r_idx+1, 0:inner_grid_num+0, 1:inner_grid_num+1]
        lab_next6 = lab[r_idx+1, 1:inner_grid_num+1, 0:inner_grid_num+0]
        lab_next7 = lab[r_idx+1, 1:inner_grid_num+1, 1:inner_grid_num+1]

        lab_next_list = [
            lab_next1, lab_next2, lab_next3, lab_next4,
            lab_next5, lab_next6, lab_next7]
        lab_next_diff_list = [
            delta_E_CIE2000(base_lab, lab_next) for lab_next in lab_next_list]
        # print(lab_next_diff_list)
        # lab_nex_diff_array's shape is (g_idx, b_idx, direction)
        lab_next_diff_array = np.dstack(lab_next_diff_list)
        sum_diff_r_direction_buf.append(lab_next_diff_array)
        # break
    sum_diff_inner_rgb = np.stack((sum_diff_r_direction_buf))

    return sum_diff_inner_rgb


def main_func(calc_de2k=False):
    grid_num = 256

    if (not Path(DE2000_8BIT_FILE).exists()) or calc_de2k:
        print("re-calculation")
        sum_diff_inner_rgb = calc_delta_e_for_next_grid(grid_num=grid_num)
        np.save(DE2000_8BIT_FILE, sum_diff_inner_rgb)
        rr, gg, bb, dd = get_top_nth_index(
            data=sum_diff_inner_rgb, inner_grid_num=grid_num-1,
            direction_num=7, nth=2**20)
        np.save(DE2000_RANK_INDEX, np.array([rr, gg, bb, dd]))
    else:
        print("load save data")
        sum_diff_inner_rgb = np.load(DE2000_8BIT_FILE)
        rgbd_array = np.load(DE2000_RANK_INDEX)
        rr = rgbd_array[0]
        gg = rgbd_array[1]
        bb = rgbd_array[2]
        dd = rgbd_array[3]

    print(rr, gg, bb, dd)
    print(sum_diff_inner_rgb[rr, gg, bb, dd])
    for idx in range(400):
        print(rr[idx], gg[idx], bb[idx], dd[idx], sum_diff_inner_rgb[rr[idx], gg[idx], bb[idx], dd[idx]])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func(calc_de2k=True)
