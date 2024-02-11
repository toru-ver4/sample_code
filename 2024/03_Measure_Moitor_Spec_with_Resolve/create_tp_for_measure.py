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
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def create_tp_base_name(
        color_mask=[1, 1, 0], cv=1023/1023, patch_area_ratio=0.03):
    patch_area_ratio_int = int(patch_area_ratio*100)
    cv_int = int(cv * 1023 + 0.5)
    color_str = f"{color_mask[0]}_{color_mask[1]}_{color_mask[2]}"
    base_name = f"color-{color_str}_cv-{cv_int:04d}_"
    base_name += f"ratio-{patch_area_ratio_int:03d}"

    return base_name


def draw_cv_info(img, cv, font_size, fg_color):
    text = f"{int(cv * 1023 + 0.5):04d} CV"

    # create instance
    text_draw_ctrl = fc2.TextDrawControl(
        text=text, font_color=fg_color,
        font_size=font_size, font_path=fc2.NOTO_SANS_CJKJP_MEDIUM,
        stroke_width=0, stroke_fill=None)

    # calc position
    pos = (10, 10)

    text_draw_ctrl.draw(img=img, pos=pos)


def calc_patch_size(img, patch_area_ratio):
    height, width = img.shape[:2]
    areta_all = width * height
    areta_patch = areta_all * patch_area_ratio
    size = int(areta_patch ** 0.5)

    return size


def create_patch(color_mask=[1, 1, 0], cv=1023/1023, patch_area_ratio=0.03):
    # width = 3840
    # height = 2160
    width = 1920
    height = 1080
    linear_cv = tf.eotf(cv, tf.GAMMA24)  # linearization for text rendering
    font_size = int(18 * height / 1080 + 0.5)
    font_fg_color = tf.eotf(np.array([0.1, 0.1, 0.1]), tf.GAMMA24)
    fname_base = create_tp_base_name(
        color_mask=color_mask, cv=cv, patch_area_ratio=patch_area_ratio)
    fname = f"./tp_img/{fname_base}.png"

    img = np.zeros((height, width, 3))
    size = calc_patch_size(img=img, patch_area_ratio=patch_area_ratio)
    patch = np.ones((size, size, 3)) * linear_cv * np.array(color_mask)
    pos = ((width // 2) - (size // 2), (height // 2) - (size // 2))
    tpg.merge(img, patch, pos)

    draw_cv_info(img=img, cv=cv, font_size=font_size, fg_color=font_fg_color)

    print(fname)
    tpg.img_wirte_float_as_16bit_int(
        filename=fname, img_float=tf.oetf(img, tf.GAMMA24))


def create_black():
    width = 1920
    height = 2160
    img = np.zeros((height, width, 3))
    fname = "./tp_img/black.png"
    print(fname)
    tpg.img_wirte_float_as_16bit_int(filename=fname, img_float=img)


def create_cv_list(num_of_block=64):
    step = 1024 // num_of_block
    cv_list = np.arange(0, 1024, step)
    cv_list[-1] = 1023

    return cv_list


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_black()
    # cv_list = create_cv_list(num_of_block=64)
    # print(cv_list)
    # for cv in cv_list:
    #     create_patch(color_mask=[1, 1, 1], cv=cv/1023, patch_area_ratio=0.03)
