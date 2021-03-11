# -*- coding: utf-8 -*-
"""
fix alpha blending of font_control
===================================
"""

# import standard libraries
import os
from pathlib import Path

# import third-party libraries
import numpy as np

# import my libraries
import test_pattern_generator2 as tpg
from test_pattern_coordinate import GridCoordinate

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


IMG_SEQ_DIR = Path("/work/overuse/2021/02_8bt_10bit/img_seq")


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


def id_patch_generator_class_test(
        width=512, height=512, total_step=30, level='low', step=2):
    generator = tpg.IdPatch8bit10bitGenerator(
        width=width, height=height, total_step=total_step, level=level,
        slide_step=step)
    frame_num = 180
    fname_8bit_base = "img_8bit_{width}x{height}_{step}step_{div}div_"
    fname_8bit_base += "{level}_{idx:04d}.png"
    fname_8bit_base = str(IMG_SEQ_DIR / fname_8bit_base)
    fname_10bit_base = "img_10bit_{width}x{height}_{step}step_{div}div_"
    fname_10bit_base += "{level}_{idx:04d}.png"
    fname_10bit_base = str(IMG_SEQ_DIR / fname_10bit_base)

    for idx in range(frame_num):
        img_8bit, img_10bit = generator.extract_8bit_10bit_img()
        fname_8bit = fname_8bit_base.format(
            width=width, height=height, step=step, div=total_step,
            level=level, idx=idx)
        fname_10bit = fname_10bit_base.format(
            width=width, height=height, step=step, div=total_step,
            level=level, idx=idx)
        print(fname_8bit)
        tpg.img_wirte_float_as_16bit_int(fname_8bit, img_8bit)
        tpg.img_wirte_float_as_16bit_int(fname_10bit, img_10bit)


def grid_coordinate_test(
        width=1920, height=1080, h_num=6, v_num=4,
        fg_width=200, fg_height=150, remove_tblr_margin=False):
    gc = GridCoordinate(
        bg_width=width, bg_height=height,
        fg_width=fg_width, fg_height=fg_height,
        h_num=h_num, v_num=v_num, remove_tblr_margin=remove_tblr_margin)
    pos_list = gc.get_st_pos_list()

    img = np.zeros((height, width, 3))
    fg_img = np.ones((fg_height, fg_width, 3))
    for v_idx in range(v_num):
        for h_idx in range(h_num):
            idx = v_idx * h_num + h_idx
            tpg.merge(
                img, fg_img*(idx+1)/(h_num*v_num + 1), pos_list[h_idx][v_idx])
    tpg.img_wirte_float_as_16bit_int("./test_img.png", img)


def main_func():
    # create_and_save_8bit_10bit_patch(
    #     width=512, height=512, total_step=20, direction='h', level='low')
    # create_and_save_8bit_10bit_patch(
    #     width=512, height=512, total_step=20, direction='v', level='low')
    # create_and_save_8bit_10bit_patch(
    #     width=512, height=512, total_step=20, direction='h', level='middle')
    # create_and_save_8bit_10bit_patch(
    #     width=512, height=512, total_step=20, direction='v', level='middle')
    # create_and_save_8bit_10bit_patch(
    #     width=512, height=512, total_step=20, direction='h', level='high')
    # create_and_save_8bit_10bit_patch(
    #     width=512, height=512, total_step=20, direction='v', level='high')
    # id_patch_generator_class_test(
    #     width=512, height=512, total_step=20, level=tpg.L_HIGH_C_HIGH, step=4)
    # grid_coordinate_test()
    grid_coordinate_test(
        width=200, height=1080, h_num=1, v_num=3,
        fg_width=200, fg_height=150, remove_tblr_margin=True)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
