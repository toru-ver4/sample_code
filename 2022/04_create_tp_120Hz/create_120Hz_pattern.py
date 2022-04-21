# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from turtle import st

# import third-party libraries
import numpy as np
from colour.io import write_image
from sympy import div

# import my libraries
import font_control2 as fc2
import test_pattern_generator2 as tpg
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


FFMPEG_NORMALIZE_COEF = 65340
FFMPEG_NORMALIZE_COEF_INV = 65535/65340

REVISION = 1


def main_func():
    pass


def complex_dot_pattern2(
        nn=3,
        fg_color=np.array([1.0, 0.5, 0.3]),
        bg_color=np.array([0.1, 0.1, 0.1]),
        bg_color_alpha=0.0):
    """
    Parameters
    ----------
    nn : int
        factor N !!!!
    fg_color : ndarray
        color value. It must be linear.
    bg_color : ndarray
        color value. It must be linear.
    bg_color_alpha : ndarray

    """
    if nn < 1:
        img = np.ones((1, 1, 3)) * fg_color
    elif nn == 1:
        img = np.ones((2, 2, 3)) * fg_color
        img[0, 0] = bg_color
        img[1, 1] = bg_color
    else:
        size = 2 ** nn
        div4 = size // 4
        # div8 = size // 8
        pt1 = (div4 * 0, div4 * 0)
        pt2 = (div4 * 3, div4 * 0)
        pt3 = (div4 * 2, div4 * 1)
        pt4 = (div4 * 4, div4 * 1)
        pt5 = (div4 * 1, div4 * 2)
        pt6 = (div4 * 2, div4 * 2)
        pt7 = (div4 * 3, div4 * 2)
        pt8 = (div4 * 0, div4 * 3)
        pt9 = (div4 * 2, div4 * 3)
        pt10 = (div4 * 1, div4 * 4)
        pt11 = (div4 * 4, div4 * 4)

        # pt12 = (div8 * 4, div8 * 0)
        # pt13 = (div8 * 5, div8 * 1)
        # pt14 = (div8 * 6, div8 * 2)
        # pt15 = (div8 * 7, div8 * 3)
        # pt16 = (div8 * 0, div8 * 4)
        # pt17 = (div8 * 8, div8 * 4)
        # pt18 = (div8 * 1, div8 * 5)
        # pt19 = (div8 * 2, div8 * 6)
        # pt20 = (div8 * 3, div8 * 7)
        # pt21 = (div8 * 4, div8 * 8)

        img = np.ones((size, size, 3)) * bg_color
        img_n1 = complex_dot_pattern2(
            nn=nn-1, fg_color=fg_color, bg_color=bg_color,
            bg_color_alpha=bg_color_alpha)
        img_n2 = complex_dot_pattern2(
            nn=nn-2, fg_color=fg_color, bg_color=bg_color,
            bg_color_alpha=bg_color_alpha)
        # img_n3 = complex_dot_pattern2(
        #     nn=nn-3, fg_color=fg_color, bg_color=bg_color,
        #     bg_color_alpha=bg_color_alpha)

        img[pt1[1]:pt6[1], pt1[0]:pt6[0]] = fg_color
        img[pt2[1]:pt4[1], pt2[0]:pt4[0]] = img_n2[:, ::-1, :]
        img[pt8[1]:pt10[1], pt8[0]:pt10[0]] = img_n2[::-1, :, :]
        img[pt3[1]:pt7[1], pt3[0]:pt7[0]] = img_n2[:, ::-1, :]
        img[pt5[1]:pt9[1], pt5[0]:pt9[0]] = img_n2[::-1, :, :]
        img[pt6[1]:pt11[1], pt6[0]:pt11[0]] = img_n1

        # img[pt12[1]:pt13[1], pt12[0]:pt13[0]] = img_n3
        # img[pt13[1]:pt14[1], pt13[0]:pt14[0]] = img_n3
        # img[pt14[1]:pt15[1], pt14[0]:pt15[0]] = img_n3
        # img[pt15[1]:pt17[1], pt15[0]:pt17[0]] = img_n3

        # img[pt16[1]:pt18[1], pt16[0]:pt18[0]] = img_n3
        # img[pt18[1]:pt19[1], pt18[0]:pt19[0]] = img_n3
        # img[pt19[1]:pt20[1], pt19[0]:pt20[0]] = img_n3
        # img[pt20[1]:pt21[1], pt20[0]:pt21[0]] = img_n3

    return img


def debug_dot_pattern():
    fname = "./img/org_dot.png"
    img = tpg.complex_dot_pattern(
        kind_num=4, whole_repeat=1)
    write_image(img, fname)

    nn = 8
    fg_color = np.array([1, 1, 1])
    bg_color = np.array([0, 0, 0])
    fname = f"./img/coplex_dot_n-{nn}.png"
    img = complex_dot_pattern2(
        nn=nn, fg_color=fg_color, bg_color=bg_color)
    write_image(img, fname)


def line_cross_pattern(nn, num_of_min_line, fg_color, bg_color):
    """
    nn : int
        factor N
    num_of_min_line : int
        minimum line number
    fg_color : ndarray
        color value. It must be linear.
    bg_color : ndarray
        color value. It must be linear.
    """
    max_thickness = 2 ** (nn - 1)
    block_len = max_thickness * num_of_min_line * 2
    size = block_len * nn
    print(f"block_len={block_len}, size={size}")
    img = np.ones((size, size, 3)) * bg_color

    for n_idx in range(nn):
        thickness = max_thickness // (2 ** n_idx)
        g_st_pos = [0, block_len * n_idx]
        num_of_line = num_of_min_line * (2 ** n_idx)
        for l_idx in range(num_of_line):
            st_pos = [0, g_st_pos[1] + thickness * 2 * l_idx]
            # print(f"l_idx={l_idx}, st_pos={st_pos}, thick={thickness}")
            draw_line(
                img=img, st_pos=st_pos, width=size, thickness=thickness,
                color=fg_color, direction='h')
            draw_line(
                img=img, st_pos=st_pos, width=size, thickness=thickness,
                color=fg_color, direction='v')

    img = tf.oetf(img, tf.GAMMA24)
    fname = f"./img/line_cross_nn-{nn}_nol-{num_of_min_line}.png"
    write_image(img, fname)


def draw_line(img, st_pos, width, thickness, color, direction='h'):
    if direction == 'h':
        st_pos2 = st_pos
        ed_pos = [st_pos[0] + width, st_pos[1] + thickness]
        # print(f"{st_pos2}, {ed_pos}")
    elif direction == 'v':
        st_pos2 = [st_pos[1], st_pos[0]]
        ed_pos = [st_pos2[0] + thickness, st_pos2[1] + width]
        # print(f"{st_pos2}, {ed_pos}")
    else:
        raise ValueError("invalid direction")

    draw_rectangle(img, st_pos2, ed_pos, color)


def draw_rectangle(img, st_pos, ed_pos, color):
    img[st_pos[1]:ed_pos[1], st_pos[0]:ed_pos[0]] = color


def debug_func():
    # debug_dot_pattern()
    fg_color = np.array([1, 1, 1])
    bg_color = np.array([0, 0, 0])
    line_cross_pattern(
        nn=6, num_of_min_line=1, fg_color=fg_color, bg_color=bg_color)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_func()
    # main_func()
