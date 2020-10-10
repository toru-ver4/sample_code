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
from colour import RGB_to_XYZ, XYZ_to_RGB, XYZ_to_Lab, Lab_to_XYZ,\
    RGB_COLOURSPACES
from scipy import interpolate

# import my libraries
import test_pattern_generator2 as tpg
import color_space as cs

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


def arctan_0_to_2pi(xx, yy):
    """
    xx, yy の値から HUE を計算する。
    出力の値域は [0, 2pi) である。

    Examples
    --------
    >>> aa=np.array([1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0.0, 0.5, 0.99])*np.pi,
    >>> bb=np.array([0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5, -0.001])*np.pi
    >>> hue = calc_hue_from_ab(aa, bb)
    [0.  45.  90.  135.  180.  225.  270.  315. 359.94212549]
    """
    hue = np.where(xx != 0, np.arctan(yy/xx), np.pi/2*np.sign(yy))
    add_pi_idx = (xx < 0) & (yy >= 0)
    sub_pi_idx = (xx < 0) & (yy < 0)
    hue[add_pi_idx] = hue[add_pi_idx] + np.pi
    hue[sub_pi_idx] = hue[sub_pi_idx] - np.pi

    hue[hue < 0] = hue[hue < 0] + 2 * np.pi

    return hue


def calc_lxy(img, p_pos, q_pos):
    """
    calc gradation value.
    """
    xx, yy = calc_xy_coordinate(img)
    pq_x = (q_pos[0] - p_pos[0])
    pq_y = (q_pos[1] - p_pos[1])
    rad_pq = arctan_0_to_2pi(xx=np.array([pq_x]), yy=np.array([pq_y]))[0]

    pr_x = (xx - p_pos[0])
    pr_y = (yy - p_pos[1])
    rad_r1 = arctan_0_to_2pi(xx=pr_x, yy=pr_y)

    rad_r2 = rad_r1 - rad_pq
    l_pr = ((xx - p_pos[0]) ** 2 + (yy - p_pos[1]) ** 2) ** 0.5
    l_xy = l_pr * np.cos(rad_r2)
    l_xy = (l_xy - np.min(l_xy)) / (np.max(l_xy) - np.min(l_xy))
    l_xy = np.clip(l_xy, 0.0, 1.0)

    return l_xy.reshape((img.shape[0], img.shape[1], 1))


def calc_lut_cielab_func(
        color_1=np.array([1.0, 1.0, 0.0]),
        color_2=np.array([0.0, 0.0, 1.0])):
    cielab_0 = XYZ_to_Lab(
        RGB_to_XYZ(
            color_1, cs.D65, cs.D65,
            RGB_COLOURSPACES[cs.BT709].RGB_to_XYZ_matrix))
    cielab_2 = XYZ_to_Lab(
        RGB_to_XYZ(
            color_2, cs.D65, cs.D65,
            RGB_COLOURSPACES[cs.BT709].RGB_to_XYZ_matrix))

    middle_ll = (cielab_0[0] + cielab_2[0]) / 2
    cielab_1 = np.array([middle_ll, 0.0, 0.0])

    lut = np.array([cielab_0, cielab_1, cielab_2])

    zz = np.linspace(0, 1, lut.shape[0])
    func_lab = [interpolate.interp1d(zz, lut[..., idx]) for idx in range(3)]

    return func_lab


def calc_point_p_q(img, rate):
    height = img.shape[0]
    width = img.shape[1]

    pos_0 = (0, 0)
    pos_1 = (0, height - 1)
    pos_2 = (width - 1, height - 1)
    pos_3 = (width - 1, 0)
    pos_4 = (0, 0)

    pos = np.array([pos_0, pos_1, pos_2, pos_3, pos_4])

    zz = [
        0,
        height - 1,
        height - 1 + width - 1,
        height - 1 + width - 1 + height - 1,
        height - 1 + width - 1 + height - 1 + width - 1
    ]

    func_pos = [interpolate.interp1d(zz, pos[..., idx]) for idx in range(2)]
    total_num = ((height - 1) * 2 + (width - 1) * 2)
    rate2 = rate * total_num
    rate3 = rate + 0.5 - int(rate + 0.5)
    rate3 = rate3 * total_num
    p_pos_x = int(func_pos[0](rate2) + 0.5)
    p_pos_y = int(func_pos[1](rate2) + 0.5)
    q_pos_x = int(func_pos[0](rate3) + 0.5)
    q_pos_y = int(func_pos[1](rate3) + 0.5)

    return (p_pos_x, p_pos_y), (q_pos_x, q_pos_y)


def change_bit_depth(img, bit_depth=8):
    img_10bit = np.uint16(np.round(img * 1023))
    mask = (2 ** bit_depth - 1) << (10 - bit_depth)
    print(f"mask = 0x{mask:04X}")
    img_n_bit = img_10bit & mask

    return img_n_bit / 1023


def gradation_rate_debug():
    width = 900
    height = 900
    gamma = 2.0
    # y_rate = 0.5
    # p_pos = (0, height - 1 - (int(height * y_rate)))
    # q_pos = (width - 1, height - 1 - p_pos[1])
    for idx, rate in enumerate(np.linspace(0, 1, 24)):
        img = np.ones((height, width, 3))
        p_pos, q_pos = calc_point_p_q(img, rate)

        l_xy = calc_lxy(img, p_pos, q_pos)
        img = img * l_xy

        func_lab = calc_lut_cielab_func(
            color_1=np.array([0.0, 0.0, 1.0]),
            color_2=np.array([1.0, 1.0, 0.0]))

        lab_value = np.dstack([func(l_xy) for func in func_lab])
        rgb_linear = XYZ_to_RGB(
            Lab_to_XYZ(lab_value), cs.D65, cs.D65,
            RGB_COLOURSPACES[cs.BT709].XYZ_to_RGB_matrix)
        rgb_linear = np.clip(rgb_linear, 0.0, 1.0)
        rgb = rgb_linear ** (1/gamma)
        img = rgb

        cv2.circle(img, p_pos, 1, (32/255, 32/255, 32/255), -1)
        cv2.circle(img, q_pos, 1, (235/255, 235/255, 235/255), -1)

        img_6bit = change_bit_depth(img, bit_depth=6)
        img_8bit = change_bit_depth(img, bit_depth=8)

        # img = cv2.resize(
        #     img, None, fx=30, fy=30, interpolation=cv2.INTER_NEAREST)
        fname_6 = "/work/overuse/2020/028_8bit_10bit_distinguish_tp/img_seq/"
        fname_6 += f"yb_img_6bit{idx:04d}.png"
        fname_8 = "/work/overuse/2020/028_8bit_10bit_distinguish_tp/img_seq/"
        fname_8 += f"yb_img_8bit{idx:04d}.png"
        tpg.img_wirte_float_as_16bit_int(fname_6, img_6bit)
        tpg.img_wirte_float_as_16bit_int(fname_8, img_8bit)


def experimental_func():
    gradation_rate_debug()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    experimental_func()
