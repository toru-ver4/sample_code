# -*- coding: utf-8 -*-

# import standard libraries
import sys
import os
from pathlib import Path

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt

# import my libraries
import test_pattern_generator2 as tpg

def calc_color_checker_pattern_block_st_pos(
        color_idx=1, width=1920, grey_block_size=32, color_block_size=64):
    color_checker_h_num = 6
    block_num_h = calc_block_num_h(width=width, block_size=grey_block_size)
    st_pos_v_offset = ((1023 // block_num_h) + 8) * grey_block_size
    st_pos_v = (color_idx // color_checker_h_num) * color_block_size\
        + st_pos_v_offset
    st_pos_h = (color_idx % color_checker_h_num) * color_block_size
    st_pos = (st_pos_h, st_pos_v)

    return st_pos


def calc_rgbmyc_pattern_block_st_pos(
        color_idx=1, width=1920, grey_block_size=32, color_block_size=64):
    block_num_h = calc_block_num_h(width=width, block_size=grey_block_size)
    st_pos_v = ((1023 // block_num_h) + 4) * grey_block_size
    st_pos_h = (color_idx % block_num_h) * color_block_size
    st_pos = (st_pos_h, st_pos_v)

    return st_pos


def calc_block_num_h(width=1920, block_size=64):
    return width // block_size


def calc_gradation_pattern_block_st_pos(code_value, width, block_size):
    block_num_h = calc_block_num_h(width=width, block_size=block_size)
    st_pos_h = (code_value % block_num_h) * block_size
    st_pos_v = (code_value // block_num_h) * block_size
    st_pos = (st_pos_h, st_pos_v)

    return st_pos


def create_10bit_pattern_for_full_range_encode():
    output_fname = "./img/src_img.png"
    width = 1920
    height = 1080
    cv_max = 1023
    grey_patch_size = 32
    color_patch_size = 64
    img = np.zeros((height, width, 3), dtype=np.uint16)
    grey_block_img_base = np.ones((grey_patch_size, grey_patch_size, 3), dtype=np.uint16)
    color_block_img_base = np.ones((color_patch_size, color_patch_size, 3), dtype=np.uint16)

    for cv in range(cv_max+1):
        block_img = grey_block_img_base * cv
        st_pos = calc_gradation_pattern_block_st_pos(
            code_value=cv, width=width, block_size=grey_patch_size
        )
        tpg.merge(img, block_img, st_pos)

    # RGBMYC
    color_list = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1]],
        dtype=np.uint8
    )
    for color_idx in range(len(color_list)):
        block_img = color_block_img_base * color_list[color_idx] * cv_max
        st_pos = calc_rgbmyc_pattern_block_st_pos(
            color_idx=color_idx, width=width,
            grey_block_size=grey_patch_size, color_block_size=color_patch_size
        )
        tpg.merge(img, block_img, st_pos)

    # Color Checker
    color_checker_linear = tpg.generate_color_checker_rgb_value()
    rgb_value_gm24 = np.clip(color_checker_linear, 0.0, 1.0) ** (1/2.4)
    rgb_value_10bit = np.uint16(np.round(rgb_value_gm24 * cv_max))

    for color_idx in range(len(rgb_value_10bit)):
        block_img = color_block_img_base * rgb_value_10bit[color_idx]
        st_pos = calc_color_checker_pattern_block_st_pos(
            color_idx=color_idx, width=width,
            grey_block_size=grey_patch_size, color_block_size=color_patch_size
        )
        tpg.merge(img, block_img, st_pos)

    tpg.img_wirte_float_as_16bit_int(
        filename=output_fname, img_float=img/1023
    )


#####################
# Main
#####################
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_10bit_pattern_for_full_range_encode()
