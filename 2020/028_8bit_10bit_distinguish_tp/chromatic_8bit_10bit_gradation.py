# -*- coding: utf-8 -*-
"""
create a test pattern to distinguish between 8bit and 10bit
===========================================================

Description.

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
import cv2

# import my libraries
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def calc_xy_coordinate(img):
    height = img.shape[0]
    width = img.shape[1]
    x = np.arange(width)
    y = np.arange(height)

    xx, yy = np.meshgrid(x, y)

    return xx, yy


def calc_lxy(img, p_pos, q_pos):
    """
    """
    xx, yy = calc_xy_coordinate(img)
    if q_pos[0] != p_pos[0]:
        rad_pq = np.arctan((q_pos[1] - p_pos[1]) / (q_pos[0] - p_pos[0]))
    elif q_pos[1] > p_pos[1]:
        rad_pq = np.pi / 2
    else:
        rad_pq = -np.pi / 2

    rad_r1 = np.arctan((yy - p_pos[1]) / (xx - p_pos[0]))
    upper_idx = ((xx - p_pos[0]) == 0) & (yy > p_pos[1])
    lower_idx = ((xx - p_pos[0]) == 0) & (yy <= p_pos[1])
    rad_r1[upper_idx] = np.pi / 2
    rad_r1[lower_idx] = -np.pi / 2

    rad_r2 = rad_r1 - rad_pq
    l_pq = ((q_pos[0] - p_pos[0]) ** 2 + (q_pos[1] - p_pos[1]) ** 2) ** 0.5
    l_pr = ((xx - p_pos[0]) ** 2 + (yy - p_pos[1]) ** 2) ** 0.5
    l_xy = l_pr * np.cos(rad_r2)

    return l_xy.reshape((img.shape[0], img.shape[1], 1)) / l_pq


def gradation_rate_debug():
    width = 16 * 2
    height = 9 * 2
    y_rate = 0.0
    p_pos = (0, height - 1 - (int(height * y_rate)))
    q_pos = (width - 1, height - 1 - p_pos[1])
    img = np.ones((height, width, 3))

    l_xy = calc_lxy(img, p_pos, q_pos)
    img = img * l_xy

    cv2.circle(img, p_pos, 1, (0, 255, 0), -1)
    cv2.circle(img, q_pos, 1, (255, 0, 0), -1)

    img = cv2.resize(img, None, fx=30, fy=30, interpolation=cv2.INTER_NEAREST)
    tpg.img_wirte_float_as_16bit_int("./sample.png", img)


def experimental_func():
    gradation_rate_debug()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    experimental_func()
