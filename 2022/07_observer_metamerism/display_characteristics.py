# -*- coding: utf-8 -*-
"""
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour.io import write_image

# import my libraries
import transfer_functions as tf
import font_control2 as fc2

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def add_text_upper_left(img, text):
    text_draw_ctrl = fc2.TextDrawControl(
        text=text, font_color=[0.2, 0.2, 0.2],
        font_size=50, font_path=fc2.NOTO_SANS_CJKJP_MEDIUM,
        stroke_width=5, stroke_fill=[0.0, 0.0, 0.0])

    # calc position
    _, text_height = text_draw_ctrl.get_text_width_height()
    pos_h = 0
    pos_v = (text_height // 2)
    pos = (pos_h, pos_v)

    text_draw_ctrl.draw(img=img, pos=pos)


def create_measure_patch():
    width = 1920
    height = 1080
    sample_num = 5
    color_mask_list = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    color_info_list = ["R", "G", "B", "W"]
    val_list = np.linspace(0, 1, sample_num)
    for color_idx in range(len(color_mask_list)):
        base_img = np.ones((height, width, 3))
        for val in val_list:
            img = base_img * val * color_mask_list[color_idx]
            cv_8bit = np.array([val, val, val]) * color_mask_list[color_idx]
            cv_8bit = np.uint8(np.round((cv_8bit ** (1/2.2)) * 0xFF))
            text = f" (R, G, B) = ({cv_8bit[0]}, {cv_8bit[1]}, {cv_8bit[2]})"
            add_text_upper_left(img=img, text=text)

            color_name = color_info_list[color_idx]
            fname = f"./img_measure_patch/display_measure_patch_{color_name}_"
            fname += f"{np.max(cv_8bit):03d}.png"
            print(fname)
            write_image(img**(1/2.2), fname, 'uint8')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_measure_patch()
