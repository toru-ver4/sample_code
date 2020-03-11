# -*- coding: utf-8 -*-
"""
make a checkerboard that reverses color over time
==============

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2

# import my libraries
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def make_reverse_checker_board(
        width=1920, height=1080, fps=60, second=3, checker_factor=1):
    frame_num = fps * second
    code_value_list = tpg.get_accelerated_x_4x(frame_num)
    h_tile_num = 16 * checker_factor
    v_tile_num = 9 * checker_factor
    rate = 770 / 1023 * 0xFFFF

    for idx in range(frame_num):
        value_a = code_value_list[idx]
        value_b = 1.0 - value_a

        value_a_10bit = int(value_a * rate + 0.5)
        value_b_10bit = int(value_b * rate + 0.5)
        img = tpg.make_tile_pattern(
            width=width, height=height,
            h_tile_num=h_tile_num, v_tile_num=v_tile_num,
            low_level=value_a_10bit, high_level=value_b_10bit)
        fname = f"./sequence/{width}x{height}_{fps}p_{second}s_"\
            + f"{checker_factor}x_tile_{h_tile_num}x{v_tile_num}_{idx:04d}"\
            + ".tiff"
        cv2.imwrite(fname, np.uint16(img))

    for idx in range(frame_num):
        value_a = code_value_list[idx]
        value_b = 1.0 - value_a

        value_a_10bit = int(value_a * rate + 0.5)
        value_b_10bit = int(value_b * rate + 0.5)
        img = tpg.make_tile_pattern(
            width=width, height=height,
            h_tile_num=h_tile_num, v_tile_num=v_tile_num,
            low_level=value_b_10bit, high_level=value_a_10bit)
        fname = f"./sequence/{width}x{height}_{fps}p_{second}s_"\
            + f"{checker_factor}x_tile_{h_tile_num}x{v_tile_num}_"\
            + f"{idx+frame_num:04d}.tiff"
        cv2.imwrite(fname, np.uint16(img))


def main_func():
    make_reverse_checker_board()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
