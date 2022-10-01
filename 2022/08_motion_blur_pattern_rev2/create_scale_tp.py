# -*- coding: utf-8 -*-
"""
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np

# import my libraries
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

NUM_OF_HORIZONTAL_UNIT = 16
NUM_OF_VERTICAL_UNIT = 9
NUM_OF_MINOR_SCALE = 10


def calc_horizontal_unit_size(width):
    return width // NUM_OF_HORIZONTAL_UNIT


def calc_vertical_unit_size(height):
    return height // NUM_OF_VERTICAL_UNIT


def calc_horizontal_minor_scale_size(width):
    unit_size = calc_horizontal_unit_size(width=width)
    minor_scale_size = unit_size // NUM_OF_MINOR_SCALE

    return minor_scale_size


def calc_vertical_minor_scale_size(height):
    unit_size = calc_vertical_unit_size(height=height)
    minor_scale_size = unit_size // NUM_OF_MINOR_SCALE

    return minor_scale_size


def calc_scale_major_pos_list_v(height):
    unit_size = calc_vertical_unit_size(height=height)

    x = np.arange(NUM_OF_VERTICAL_UNIT)
    major_pos_list_v = x * unit_size

    return major_pos_list_v


def calc_scale_major_pos_list_h(width):
    unit_size = calc_horizontal_unit_size(width=width)

    x = np.arange(NUM_OF_HORIZONTAL_UNIT)
    major_pos_list_h = x * unit_size

    return major_pos_list_h


def calc_scale_minor_pos_list_v(height):
    minor_scale_size = calc_vertical_minor_scale_size(height=height)
    major_pos_list = calc_scale_major_pos_list_v(height=height)
    out_buf = []
    for major_pos in major_pos_list:
        for idx in range(NUM_OF_MINOR_SCALE-1):
            pos = major_pos + minor_scale_size * (idx + 1)
            out_buf.append(pos)

    return np.array(out_buf, dtype=np.uint16)


def calc_scale_minor_pos_list_h(width):
    minor_scale_size = calc_horizontal_minor_scale_size(width=width)
    major_pos_list = calc_scale_major_pos_list_h(width=width)
    out_buf = []
    for major_pos in major_pos_list:
        for idx in range(NUM_OF_MINOR_SCALE-1):
            pos = major_pos + minor_scale_size * (idx + 1)
            out_buf.append(pos)

    return np.array(out_buf, dtype=np.uint16)


def draw_major_auxiliary_line(img, line_color):
    height, width = img.shape[:2]
    pos_list_h = calc_scale_major_pos_list_h(width=width)
    pos_list_v = calc_scale_major_pos_list_v(height=height)

    img[pos_list_v] = line_color
    img[pos_list_v+1] = line_color

    img[:, pos_list_h] = line_color
    img[:, pos_list_h+1] = line_color


def draw_minor_auxiliary_line(img, line_color):
    height, width = img.shape[:2]
    pos_list_h = calc_scale_minor_pos_list_h(width=width)
    pos_list_v = calc_scale_minor_pos_list_v(height=height)

    img[pos_list_v] = line_color
    img[:, pos_list_h] = line_color


def draw_horizontal_scale(
        img, scale_color,
        major_scale_height_rate, minor_scale_height_rate):
    height, width = img.shape[:2]

    major_height = int(major_scale_height_rate * height + 0.5)
    minor_height = int(minor_scale_height_rate * height + 0.5)

    major_pos_list = calc_scale_major_pos_list_h(width=width)
    minor_pos_list = calc_scale_minor_pos_list_h(width=width)

    img[:major_height, major_pos_list] = scale_color
    img[:major_height, major_pos_list+1] = scale_color
    img[:minor_height, minor_pos_list] = scale_color

    img[-major_height:, major_pos_list] = scale_color
    img[-major_height:, major_pos_list+1] = scale_color
    img[-minor_height:, minor_pos_list] = scale_color


def draw_vertical_scale(
        img, scale_color,
        major_scale_height_rate, minor_scale_height_rate):
    height, width = img.shape[:2]

    major_height = int(major_scale_height_rate * height + 0.5)
    minor_height = int(minor_scale_height_rate * height + 0.5)

    major_pos_list = calc_scale_major_pos_list_v(height=height)
    minor_pos_list = calc_scale_minor_pos_list_v(height=height)

    img[major_pos_list, :major_height] = scale_color
    img[major_pos_list+1, :major_height] = scale_color
    img[minor_pos_list, :minor_height] = scale_color

    img[major_pos_list, -major_height:] = scale_color
    img[major_pos_list+1, -major_height:] = scale_color
    img[minor_pos_list, -minor_height:] = scale_color


def create_scale_pattern(width=1920, height=1080):
    bg_color = np.array([0.12, 0.12, 0.12])
    scale_color = np.array([0.8, 0.8, 0.8])
    major_auxiliary_line_color = np.array([0.25, 0.25, 0.25])
    minor_auxiliary_line_color = np.array([0.18, 0.18, 0.18])
    major_scale_height_rate = 0.02
    minor_scale_height_rate = 0.00666
    img = np.ones((height, width, 3)) * bg_color

    draw_major_auxiliary_line(
        img=img, line_color=major_auxiliary_line_color)
    draw_minor_auxiliary_line(
        img=img, line_color=minor_auxiliary_line_color)

    draw_horizontal_scale(
        img=img, scale_color=scale_color,
        major_scale_height_rate=major_scale_height_rate,
        minor_scale_height_rate=minor_scale_height_rate)
    draw_vertical_scale(
        img=img, scale_color=scale_color,
        major_scale_height_rate=major_scale_height_rate,
        minor_scale_height_rate=minor_scale_height_rate)

    img_non_linear = np.clip(img, 0.0, 1.0) ** (1/2.4)
    out_fname = f"./img/scale_tp_{width}x{height}.png"
    print(out_fname)
    tpg.img_wirte_float_as_16bit_int(out_fname, img_non_linear)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_scale_pattern()
