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


def create_and_save_8bit_10bit_patch(
        width=512, height=1024, total_step=20, direction='h', level='middle'):
    img_out_8bit, img_out_10bit = tpg.create_8bit_10bit_id_patch(
        width=width, height=height, total_step=total_step,
        direction=direction, level=level)

    save_8bit_10bit_patch(
        img_8bit=img_out_8bit, img_10bit=img_out_10bit,
        width=width, height=height, total_step=total_step,
        direction=direction, level=level)


def save_8bit_10bit_patch(
        img_8bit, img_10bit,
        width=512, height=1024, total_step=20, direction='h', level='middle'):

    name_8bit\
        = f"./img/8bit_grad_{width}x{height}_dir_{direction}_{level}.png"
    name_10bit\
        = f"./img/10bit_grad_{width}x{height}_dir_{direction}_{level}.png"
    tpg.img_wirte_float_as_16bit_int(name_8bit, img_8bit)
    tpg.img_wirte_float_as_16bit_int(name_10bit, img_10bit)


def create_slide_seq(
        width=512, height=512, total_step=20, level='low'):
    img_out_8bit, img_out_10bit = tpg.create_8bit_10bit_id_patch(
        width=width, height=height, total_step=total_step,
        direction='h', level=level)


def main_func():
    create_and_save_8bit_10bit_patch(
        width=512, height=512, total_step=20, direction='h', level='low')
    create_and_save_8bit_10bit_patch(
        width=512, height=512, total_step=20, direction='v', level='low')
    create_and_save_8bit_10bit_patch(
        width=512, height=512, total_step=20, direction='h', level='middle')
    create_and_save_8bit_10bit_patch(
        width=512, height=512, total_step=20, direction='v', level='middle')
    create_and_save_8bit_10bit_patch(
        width=512, height=512, total_step=20, direction='h', level='high')
    create_and_save_8bit_10bit_patch(
        width=512, height=512, total_step=20, direction='v', level='high')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
