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
from numpy.core.defchararray import center

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
RGBMYC_COLOR_LIST = np.array(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1],
     [1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=np.uint8)
COLOR_CHECKER_LINEAR = tpg.generate_color_checker_rgb_value(
    color_space=BT709_COLOURSPACE)
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
    color_list = RGBMYC_COLOR_LIST
    for color_idx in range(len(color_list)):
        block_img = block_img_base * color_list[color_idx] * MAX_CODE_VALUE
        st_pos = calc_rgbmyc_pattern_block_st_pos(
            color_idx=color_idx, width=width, height=height,
            block_size=block_size)
        tpg.merge(img, block_img, st_pos)

    # Color Checker
    color_checker_linear_value = COLOR_CHECKER_LINEAR
    rgb_value_gm24 = np.clip(color_checker_linear_value, 0.0, 1.0) ** (1/2.4)
    rgb_value_8bit = np.uint8(np.round(rgb_value_gm24 * MAX_CODE_VALUE))

    for color_idx in range(len(rgb_value_8bit)):
        block_img = block_img_base * rgb_value_8bit[color_idx]
        st_pos = calc_color_checker_pattern_block_st_pos(
            color_idx=color_idx, width=width, height=height,
            block_size=block_size)
        tpg.merge(img, block_img, st_pos)

    return img


def make_src_tp_base_name():
    fname = "{src_png_dir}/src_grad_tp_{width}x{height}"
    fname += "_b-size_{block_size}_{frame_idx:04d}.png"

    return fname


def create_gradation_pattern_sequence(
        width=1920, height=1080, block_size=64):
    for frame_idx in range(TOTAL_FRAME):
        img = create_8bit_cms_test_pattern(
            width=width, height=height, block_size=block_size)
        fname = make_src_tp_base_name().format(
            src_png_dir=SRC_PNG_DIR, width=width, height=height,
            block_size=block_size, frame_idx=frame_idx)
        # fname = f"{SRC_PNG_DIR}/src_grad_tp_{width}x{height}"
        # fname += f"_b-size_{block_size}_{frame_idx:04d}.png"
        cv2.imwrite(fname, img[..., ::-1])


def get_specific_pos_value(img, pos):
    """
    Parameters
    ----------
    img : ndarray
        image data.
    pos : list
        pos[0] is horizontal coordinate, pos[1] is verical coordinate.
    """
    return img[pos[1], pos[0]]


def read_code_value_from_gradation_pattern(
        fname=None, width=1920, height=1080, block_size=64):
    """
    Example
    -------
    >>> read_code_value_from_gradation_pattern(
    ...     fname="./data.png, width=1920, height=1080, block_size=64)
    {'ramp': array(
          [[[  0,   0,   0],
            [  1,   1,   1],
            [  2,   2,   2],
            [  3,   3,   3],
            ...
            [252, 252, 252],
            [253, 253, 253],
            [254, 254, 254],
            [255, 255, 255]]], dtype=uint8),
     'rgbmyc': array(
           [[255,   0,   0],
            [  0, 255,   0],
            [  0,   0, 255],
            [255,   0, 255],
            [255, 255,   0],
            [  0, 255, 255]], dtype=uint8),
     'colorchecker': array(
           [[123,  90,  77],
            [201, 153, 136],
            [ 99, 129, 161],
            [ 98, 115,  75],
            ...
            [166, 168, 168],
            [128, 128, 129],
            [ 91,  93,  95],
            [ 59,  60,  61]], dtype=uint8)}
    """
    # Gradation
    print(f"reading {fname}")
    img = cv2.imread(fname, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[:, :, ::-1]
    block_offset = block_size // 2
    ramp_value = np.zeros((1, CODE_VALUE_NUM, 3), dtype=np.uint8)
    for code_value in range(CODE_VALUE_NUM):
        st_pos = calc_gradation_pattern_block_st_pos(
            code_value=code_value, width=width, height=height,
            block_size=block_size)
        center_pos = (st_pos[0] + block_offset, st_pos[1] + block_offset)
        ramp_value[0, code_value] = get_specific_pos_value(img, center_pos)

    # RGBMYC
    rgbmyc_value = np.zeros_like(RGBMYC_COLOR_LIST)
    for color_idx in range(len(RGBMYC_COLOR_LIST)):
        st_pos = calc_rgbmyc_pattern_block_st_pos(
            color_idx=color_idx, width=width, height=height,
            block_size=block_size)
        center_pos = (st_pos[0] + block_offset, st_pos[1] + block_offset)
        rgbmyc_value[color_idx] = get_specific_pos_value(img, center_pos)
    rgbmyc_value.reshape((1, rgbmyc_value.shape[0], rgbmyc_value.shape[1]))

    # ColorChecker
    color_checker_value = np.zeros_like(COLOR_CHECKER_LINEAR, dtype=np.uint8)
    for color_idx in range(len(COLOR_CHECKER_LINEAR)):
        st_pos = calc_color_checker_pattern_block_st_pos(
            color_idx=color_idx, width=width, height=height,
            block_size=block_size)
        center_pos = (st_pos[0] + block_offset, st_pos[1] + block_offset)
        color_checker_value[color_idx] = get_specific_pos_value(
            img, center_pos)
    color_checker_value.reshape(
        (1, color_checker_value.shape[0], color_checker_value.shape[1]))

    return dict(
        ramp=ramp_value, rgbmyc=rgbmyc_value,
        colorchecker=color_checker_value)


def check_src_tp_code_value(width=1920, height=1080, block_size=64):
    fname = make_src_tp_base_name().format(
        src_png_dir=SRC_PNG_DIR, width=width, height=height,
        block_size=block_size, frame_idx=TOTAL_FRAME//2)
    code_value_data = read_code_value_from_gradation_pattern(
        fname=fname, width=width, height=height, block_size=block_size)

    ramp = code_value_data['ramp']
    rgbmyc = code_value_data['rgbmyc']
    color_checker = code_value_data['colorchecker']

    # Ramp
    x = np.arange(CODE_VALUE_NUM).astype(np.uint8)
    ramp_expected = np.dstack((x, x, x))
    if np.array_equal(ramp, ramp_expected):
        print("read data matched")
    else:
        raise ValueError("read data did not match!")


def debug_func():
    width = 1920
    height = 1080
    block_size = 64
    check_src_tp_code_value(
        width=width, height=height, block_size=block_size)


def main_func():
    width = 1920
    height = 1080
    block_size = 64
    # create_gradation_pattern_sequence(
    #     width=width, height=height, block_size=block_size)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # main_func()
    debug_func()
