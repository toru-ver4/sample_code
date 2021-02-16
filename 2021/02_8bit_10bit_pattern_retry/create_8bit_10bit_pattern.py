# -*- coding: utf-8 -*-
"""
fix alpha blending of font_control
===================================
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np

# import my libraries
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


BASE_RGB_8BIT_LOW = np.array([75, 56, 33])
BASE_RGB_8BIT_MIDDLE = np.array([123, 98, 74])
BASE_RGB_8BIT_HIGH = np.array([172, 146, 119])
GREEN_DIFF = 10


def create_8bit_10bit_patch(
        width=512, height=1024, total_step=20, direction='h'):
    base_gg = BASE_RGB_8BIT_MIDDLE[1]
    rr = BASE_RGB_8BIT_MIDDLE[0]
    bb = BASE_RGB_8BIT_MIDDLE[2]

    gg_min = base_gg - (total_step // 2)
    gg_max = base_gg + (total_step // 2)

    if direction == 'h':
        patch_len = width
    else:
        patch_len = height

    gg_grad = np.linspace(gg_min, gg_max, patch_len)
    rr_static = np.ones_like(gg_grad) * rr
    bb_static = np.ones_like(gg_grad) * bb
    line = np.dstack((rr_static, gg_grad, bb_static))

    if direction == 'h':
        img_base_8bit_float = line * np.ones((height, 1, 3))
    else:
        line = line.reshape((height, 1, 3))
        img_base_8bit_float = line * np.ones((1, width, 3))

    img_out_float_8bit = img_base_8bit_float / 255
    img_out_8bit = np.round(img_out_float_8bit * 255) / 255
    img_out_10bit = np.round(img_out_float_8bit * 1023) / 1023

    name_8bit = f"./img/8bit_grad_{width}x{height}_dir_{direction}.png"
    name_10bit = f"./img/10bit_grad_{width}x{height}_dir_{direction}.png"
    tpg.img_wirte_float_as_16bit_int(name_8bit, img_out_8bit)
    tpg.img_wirte_float_as_16bit_int(name_10bit, img_out_10bit)


def main_func():
    create_8bit_10bit_patch(
        width=512, height=1024, total_step=20, direction='h')
    create_8bit_10bit_patch(
        width=1024, height=512, total_step=20, direction='v')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
