# -*- coding: utf-8 -*-
"""
fix alpha blending of font_control
===================================
"""

# import standard libraries
from operator import sub
import os

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


def bt709_to_lab(rgb_gm24):
    rgb_linear = rgb_gm24 ** 2.4
    large_xyz = RGB_to_XYZ(
        rgb_linear, cs.D65, cs.D65, BT709_COLOURSPACE.RGB_to_XYZ_matrix)
    lab = XYZ_to_Lab(large_xyz)

    return lab


def main_func():
    grid_num = 7

    # calc lab
    rgb_gm24 = LUT3D.linear_table(size=grid_num)
    lab = bt709_to_lab(rgb_gm24)
    print(lab.shape)

    # XY平面は同時計算して Z方向は forループで処理する
    inner_idx_list = np.arange(1, grid_num-1)
    inner_num = len(inner_idx_list)
    sum_diff_r_direction_buf = []
    for r_idx in inner_idx_list:
        base_lab = lab[r_idx, 1:grid_num-1, 1:grid_num-1]
        lab_next1 = lab[r_idx+0, 1:grid_num-1, 0:grid_num-2]
        lab_next2 = lab[r_idx+0, 1:grid_num-1, 2:grid_num-0]
        lab_next3 = lab[r_idx+0, 0:grid_num-2, 1:grid_num-1]
        lab_next4 = lab[r_idx+0, 2:grid_num-0, 1:grid_num-1]
        lab_next5 = lab[r_idx-1, 1:grid_num-1, 1:grid_num-1]
        lab_next6 = lab[r_idx+1, 1:grid_num-1, 1:grid_num-1]

        lab_next_list = [
            lab_next1, lab_next2, lab_next3, lab_next4,
            lab_next5, lab_next6]
        lab_next_diff_2k_list = [
            delta_E_CIE2000(base_lab, lab_next) for lab_next in lab_next_list]
        sum_lab_next_diff = np.zeros_like(lab_next_diff_2k_list[0])
        for next_diff in lab_next_diff_2k_list:
            sum_lab_next_diff = sum_lab_next_diff + next_diff
        sum_diff_r_direction_buf.append(sum_lab_next_diff)
        # break
    sum_diff_inner_rgb = np.stack((sum_diff_r_direction_buf))
    print(sum_diff_inner_rgb.shape)
    max_idx_1d = sum_diff_inner_rgb.argmax()
    rr = max_idx_1d // (inner_num ** 2)
    gg = (max_idx_1d % (inner_num ** 2)) // (inner_num ** 1)
    bb = ((max_idx_1d % (inner_num ** 2)) % (inner_num ** 1)) % inner_num
    print(rr, gg, bb)
    print(sum_diff_inner_rgb[rr, gg, bb])
    print(np.max(sum_diff_inner_rgb))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
