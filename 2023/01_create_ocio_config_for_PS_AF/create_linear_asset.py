# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour.io import write_image

# import my libraries
import font_control2 as fc2

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def calc_text_pos(img, text_align, text_draw_ctrl: fc2.TextDrawControl):
    height, width = img.shape[:2]
    text_width, text_height = text_draw_ctrl.get_text_width_height()
    margin_h = int(text_height * 0.3 + 0.5)
    margin_v = int(text_height * 0.3 + 0.5)
    # "upper-left", "upper-right", "lower-left", "lower-right"
    if text_align == "upper-left":
        pos_h = margin_h
        pos_v = margin_v
    elif text_align == "upper-right":
        pos_h = width - text_width - margin_h
        pos_v = margin_v
    elif text_align == "lower-left":
        pos_h = margin_h
        pos_v = height - text_height - margin_v
    elif text_align == "lower-right":
        pos_h = width - text_width - margin_h
        pos_v = height - text_height - margin_v
    else:
        err_msg = 'Error. Invalid parameter `text_align`'
        raise ValueError(err_msg)

    return [pos_h, pos_v]


def add_text(img, val, text_align, font_size):
    # create instance
    fg_color = np.array([1.0, 1.0, 1.0])
    text = f"{val:.2f}"
    text_draw_ctrl = fc2.TextDrawControl(
        text=text, font_color=fg_color,
        font_size=font_size, font_path=fc2.NOTO_SANS_MONO_BOLD,
        stroke_width=int(font_size*0.2+0.5), stroke_fill=[0, 0, 0])

    # calc position
    pos = calc_text_pos(
        img=img, text_align=text_align, text_draw_ctrl=text_draw_ctrl)
    text_draw_ctrl.draw(img=img, pos=pos)


def make_out_fname(val=0.18):
    fname = f"./img/asset_linear-{val:.2f}.exr"
    return fname


def create_linear_asset(
        val=0.18, text_align='upper-left', size=384, font_size=40):
    """
    Parameters
    ----------
    text_align : str
        "upper-left", "upper-right", "lower-left", "lower-right"
    """
    img = np.ones((size, size, 3)) * val
    add_text(
        img=img, val=val, text_align=text_align, font_size=font_size)

    out_fname = make_out_fname(val=val)
    print(out_fname)
    write_image(img, out_fname)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    size = 384
    font_size = 40
    align_upper_left = "upper-left"
    align_upper_right = "upper-right"
    align_lower_left = "lower-left"
    align_lower_right = "lower-right"
    # create_linear_asset(
    #     val=0.10, text_align=align_upper_left, size=size, font_size=font_size)
    # create_linear_asset(
    #     val=0.08, text_align=align_upper_right, size=size, font_size=font_size)
    # create_linear_asset(
    #     val=1.00, text_align=align_lower_left, size=size, font_size=font_size)
    # create_linear_asset(
    #     val=9.00, text_align=align_lower_right, size=size, font_size=font_size)
    create_linear_asset(
        val=100, text_align=align_upper_left, size=size, font_size=font_size)
    create_linear_asset(
        val=-99, text_align=align_lower_right, size=size, font_size=font_size)
