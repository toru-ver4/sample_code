# -*- coding: utf-8 -*-
"""

==========

"""

# import standard libraries
import os
from pathlib import Path

# import third-party libraries
import numpy as np
import cv2
from colour.models import BT709_COLOURSPACE

# import my libraries
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

FPS = 24
BIT_DEPTH = 8
CODE_VALUE_NUM = (2 ** BIT_DEPTH)
MAX_CODE_VALUE = CODE_VALUE_NUM - 1
TOTAL_SEC = 1
TOTAL_FRAME = FPS * TOTAL_SEC
COLOR_CHECKER_H_HUM = 6
SRC_PNG_DIR = "/work/overuse/2020/031_cms_for_video_playback/img_seq/"
DST_MP4_DIR = "/work/overuse/2020/031_cms_for_video_playback/mp4/"
DST_PNG_DIR = "/work/overuse/2020/031_cms_for_video_playback/mp4_to_png/"


def calc_block_num_h(width=1920, block_size=64):
    return width // block_size


def calc_gradation_pattern_block_st_pos(
        code_value=MAX_CODE_VALUE, width=1920, height=1080, block_size=64):
    block_num_h = calc_block_num_h(width=width, block_size=block_size)
    st_pos_h = (code_value % block_num_h) * block_size
    st_pos_v = (code_value // block_num_h) * block_size
    st_pos = (st_pos_h, st_pos_v)

    return st_pos


def calc_rgbmyc_pattern_block_st_pos(
        color_idx=1, width=1920, height=1080, block_size=64):
    block_num_h = calc_block_num_h(width=width, block_size=block_size)
    st_pos_v = ((MAX_CODE_VALUE // block_num_h) + 2) * block_size
    st_pos_h = (color_idx % block_num_h) * block_size
    st_pos = (st_pos_h, st_pos_v)

    return st_pos


def calc_color_checker_pattern_block_st_pos(
        color_idx=1, width=1920, height=1080, block_size=64):
    block_num_h = calc_block_num_h(width=width, block_size=block_size)
    st_pos_v_offset = ((MAX_CODE_VALUE // block_num_h) + 4) * block_size
    st_pos_v = (color_idx // COLOR_CHECKER_H_HUM) * block_size\
        + st_pos_v_offset
    st_pos_h = (color_idx % COLOR_CHECKER_H_HUM) * block_size
    st_pos = (st_pos_h, st_pos_v)

    return st_pos


def create_8bit_cms_test_pattern(width=1920, height=1080, block_size=64):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    block_img_base = np.ones((block_size, block_size, 3), dtype=np.uint8)

    # gradation
    for code_value in range(CODE_VALUE_NUM):
        block_img = block_img_base * code_value
        st_pos = calc_gradation_pattern_block_st_pos(
            code_value=code_value, width=width, height=height,
            block_size=block_size)
        tpg.merge(img, block_img, st_pos)

    # RGBMYC
    color_list = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1],
         [1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    for color_idx in range(len(color_list)):
        block_img = block_img_base * color_list[color_idx] * MAX_CODE_VALUE
        st_pos = calc_rgbmyc_pattern_block_st_pos(
            color_idx=color_idx, width=width, height=height,
            block_size=block_size)
        tpg.merge(img, block_img, st_pos)

    # Color Checker
    rgb_value_float = tpg.generate_color_checker_rgb_value(
        color_space=BT709_COLOURSPACE)
    rgb_value_gm24 = np.clip(rgb_value_float, 0.0, 1.0) ** (1/2.4)
    rgb_value_8bit = np.uint8(np.round(rgb_value_gm24 * MAX_CODE_VALUE))

    for color_idx in range(len(rgb_value_8bit)):
        block_img = block_img_base * rgb_value_8bit[color_idx]
        st_pos = calc_color_checker_pattern_block_st_pos(
            color_idx=color_idx, width=width, height=height,
            block_size=block_size)
        tpg.merge(img, block_img, st_pos)

    return img


def create_gradation_pattern_sequence(
        width=1920, height=1080, block_size=64):
    for frame_idx in range(TOTAL_FRAME):
        img = create_8bit_cms_test_pattern(
            width=width, height=height, block_size=block_size)
        fname = f"{SRC_PNG_DIR}/src_grad_tp_{width}x{height}"
        fname += f"_b-size_{block_size}_{frame_idx:04d}.png"
        cv2.imwrite(fname, img[..., ::-1])


def main_func():
    width = 1920
    height = 1080
    block_size = 64
    create_gradation_pattern_sequence(
        width=width, height=height, block_size=block_size)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
