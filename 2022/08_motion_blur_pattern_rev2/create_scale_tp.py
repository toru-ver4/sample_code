# -*- coding: utf-8 -*-
"""
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np

# import my libraries
import test_pattern_generator2 as tpg
import font_control2 as fc2

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

NUM_OF_HORIZONTAL_UNIT = 16
NUM_OF_VERTICAL_UNIT = 9
NUM_OF_MINOR_SCALE = 8


def calc_horizontal_unit_size(width):
    return width // NUM_OF_HORIZONTAL_UNIT


def calc_vertical_unit_size(height):
    return height // NUM_OF_VERTICAL_UNIT


def calc_minor_scale_size(unit_size):
    minor_scale_size = unit_size // NUM_OF_MINOR_SCALE

    return minor_scale_size


def calc_horizontal_minor_scale_size(width):
    unit_size = calc_horizontal_unit_size(width=width)
    minor_scale_size = calc_minor_scale_size(unit_size=unit_size)

    return minor_scale_size


def calc_vertical_minor_scale_size(height):
    unit_size = calc_vertical_unit_size(height=height)
    minor_scale_size = calc_minor_scale_size(unit_size=unit_size)

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


def calc_major_scale_height(major_scale_height_rate, height):
    height = int(major_scale_height_rate * height + 0.5)

    return height


def draw_horizontal_scale(
        img, scale_color,
        major_scale_height_rate, minor_scale_height_rate):
    height, width = img.shape[:2]

    major_height = calc_major_scale_height(
        major_scale_height_rate=major_scale_height_rate, height=height)
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


def calc_font_size(base_font_size, height):
    font_size = base_font_size * (height // 1080)

    return font_size


def calc_font_edge_size(font_size):
    font_edge_size = int(font_size * 0.1 + 0.5)

    return font_edge_size


def draw_vertical_info_text(
        img, width, height, base_font_size, font_color, font_edge_color,
        font_path, major_scale_height_rate):
    font_size = calc_font_size(base_font_size=base_font_size, height=height)
    font_edge_size = calc_font_edge_size(font_size=font_size)

    major_pos_list = calc_scale_major_pos_list_v(height=height)
    text_list = [str(pos) for pos in major_pos_list]
    major_scale_height = calc_major_scale_height(
        major_scale_height_rate, height)

    # for text, center_pos in zip(text_list, major_pos_list):
    for idx in range(1, len(major_pos_list)):
        text = text_list[idx]
        center_pos = major_pos_list[idx]

        # left
        text_draw_ctrl = fc2.TextDrawControl(
            text=text, font_color=font_color,
            font_size=font_size, font_path=font_path,
            stroke_width=font_edge_size, stroke_fill=font_edge_color)
        text_width, text_height = text_draw_ctrl.get_text_width_height()
        h_pos = major_scale_height + int(text_height * 0.5)
        v_pos = center_pos - (text_width // 2)
        pos = (h_pos, v_pos)
        text_draw_ctrl.draw(img=img, pos=pos, rotate=270)

        # right
        text_draw_ctrl = fc2.TextDrawControl(
            text=text, font_color=font_color,
            font_size=font_size, font_path=font_path,
            stroke_width=font_edge_size, stroke_fill=font_edge_color)
        h_pos = width - h_pos - text_height
        pos = (h_pos, v_pos)
        text_draw_ctrl.draw(img=img, pos=pos, rotate=90)


def draw_horizontal_info_text(
        img, width, height, base_font_size, font_color, font_edge_color,
        font_path, major_scale_height_rate):
    font_size = calc_font_size(base_font_size=base_font_size, height=height)
    font_edge_size = calc_font_edge_size(font_size=font_size)

    major_pos_list = calc_scale_major_pos_list_h(width=width)
    text_list = [str(pos) for pos in major_pos_list]
    major_scale_height = calc_major_scale_height(
        major_scale_height_rate, height)

    # for text, center_pos in zip(text_list, major_pos_list):
    for idx in range(1, len(major_pos_list)):
        text = text_list[idx]
        center_pos = major_pos_list[idx]

        # top
        text_draw_ctrl = fc2.TextDrawControl(
            text=text, font_color=font_color,
            font_size=font_size, font_path=font_path,
            stroke_width=font_edge_size, stroke_fill=font_edge_color)
        text_width, text_height = text_draw_ctrl.get_text_width_height()
        h_pos = center_pos - (text_width // 2)
        v_pos = major_scale_height + int(text_height * 0.5)
        pos = (h_pos, v_pos)
        text_draw_ctrl.draw(img=img, pos=pos)

        # bottom
        text_draw_ctrl = fc2.TextDrawControl(
            text=text, font_color=font_color,
            font_size=font_size, font_path=font_path,
            stroke_width=font_edge_size, stroke_fill=font_edge_color)
        v_pos = height - v_pos - text_height
        pos = (h_pos, v_pos)
        text_draw_ctrl.draw(img=img, pos=pos)


def draw_center_cross(img, line_color):
    height, width = img.shape[:2]
    length = calc_vertical_minor_scale_size(height=height) * 2

    center_pos_h = width // 2
    center_pos_v = height // 2

    v_line_st_pos_h = center_pos_h - length
    v_line_ed_pos_h = center_pos_h + length + 1
    v_line_st_pos_v = center_pos_v - 1
    v_line_ed_pos_v = center_pos_v + 3

    h_line_st_pos_h = center_pos_h - 1
    h_line_ed_pos_h = center_pos_h + 3
    h_line_st_pos_v = center_pos_v - length
    h_line_ed_pos_v = center_pos_v + length + 1

    img[v_line_st_pos_v:v_line_ed_pos_v, v_line_st_pos_h:v_line_ed_pos_h]\
        = line_color
    img[h_line_st_pos_v:h_line_ed_pos_v, h_line_st_pos_h:h_line_ed_pos_h]\
        = line_color


def create_scale_pattern(width=1920, height=1080):
    bg_color = np.array([0.12, 0.12, 0.12])
    scale_color = np.array([0.6, 0.6, 0.6])
    font_color = np.array([0.6, 0.6, 0.6])
    font_edge_color = np.array([0.05, 0.05, 0.05])
    center_cross_color = np.array([0.00, 0.00, 0.00])
    major_auxiliary_line_color = np.array([0.25, 0.25, 0.25])
    minor_auxiliary_line_color = np.array([0.18, 0.18, 0.18])
    major_scale_height_rate = 0.02
    minor_scale_height_rate = 0.01
    base_font_size = 22
    font_path = fc2.NOTO_SANS_MONO_BOLD
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

    draw_horizontal_info_text(
        img=img, width=width, height=height,
        base_font_size=base_font_size,
        font_color=font_color, font_edge_color=font_edge_color,
        font_path=font_path, major_scale_height_rate=major_scale_height_rate)
    draw_vertical_info_text(
        img=img, width=width, height=height,
        base_font_size=base_font_size,
        font_color=font_color, font_edge_color=font_edge_color,
        font_path=font_path, major_scale_height_rate=major_scale_height_rate)

    draw_center_cross(img=img, line_color=center_cross_color)

    tpg.draw_outline(img=img, fg_color=scale_color, outline_width=2)

    img_non_linear = np.clip(img, 0.0, 1.0) ** (1/2.4)
    out_fname = f"./img/scale_tp_{width}x{height}.png"
    print(out_fname)
    tpg.img_wirte_float_as_16bit_int(out_fname, img_non_linear)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_scale_pattern(width=1920, height=1080)
    create_scale_pattern(width=3840, height=2160)
    create_scale_pattern(width=2048, height=1080)
    create_scale_pattern(width=4096, height=2160)
