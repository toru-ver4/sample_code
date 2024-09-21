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
import transfer_functions as tf
import test_pattern_generator2 as tpg
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def draw_text(img, text, patch_st_pos_v):
    _, width = img.shape[:2]
    font = fc2.NOTO_SANS_CJKJP_BOLD
    color_cv = tf.oetf_from_luminance(6, tf.ST2084)

    # create instance
    text_draw_ctrl = fc2.TextDrawControl(
        text=text, font_color=[color_cv, color_cv, color_cv],
        font_size=60, font_path=font,
        stroke_width=0, stroke_fill=None)

    # calc position
    text_width, text_height = text_draw_ctrl.get_text_width_height()
    pos_h = (width // 2) - (text_width // 2)
    pos_v = patch_st_pos_v - int(text_height * 1.1)
    pos = (pos_h, pos_v)

    text_draw_ctrl.draw(img=img, pos=pos)


def make_dst_heif_fname(src_png_name: str):
    heif_name = src_png_name.replace(".png", ".heic")
    heif_name = heif_name.replace("src_png", "dst_heif")

    return heif_name


def create_patch_png_image(luminance=100):
    width = 1080
    height = 1920
    patch_size = int(width * 0.65)

    cv_float = tf.oetf_from_luminance(luminance, tf.ST2084)
    cv_10bit = round(cv_float * 1023)
    text = f"{cv_10bit:04}/1023 CV, {luminance:5.3f} nits"
    
    base_img = np.zeros((height, width, 3))
    patch_img = np.ones((patch_size, patch_size, 3)) * cv_float

    patch_st_pos_h = (width//2) - (patch_size//2)
    patch_st_pos_v = (height//2) - (patch_size//2)

    tpg.merge(base_img, patch_img, pos=(patch_st_pos_h, patch_st_pos_v))

    draw_text(img=base_img, text=text, patch_st_pos_v=patch_st_pos_v)

    int_part = int(luminance)
    dec_part = luminance - int_part

    out_fname = f"./src_png/patch_{int_part:05d}{dec_part:.3f}-nits.png"
    print(out_fname)
    write_image(image=base_img, path=out_fname, bit_depth='uint16')

    return out_fname


def create_heif_luminance_patch(luminance=1000):
    png_fname = create_patch_png_image(luminance=luminance)
    heif_fname = make_dst_heif_fname(src_png_name=png_fname)
    tpg.png_to_heif(
        png_fname=png_fname, heif_fname=heif_fname,
        color_space_name=cs.BT2020,
        transfer_characteristics=tf.ST2084
    )


def create_luminance_value_array(num_of_array: int = 33):
    base_cv = 1024 // (num_of_array - 1)
    cv_list = [base_cv * idx for idx in range(num_of_array)]
    cv_list[-1] = cv_list[-1] - 1

    luminance_list = tf.eotf_to_luminance(np.array(cv_list)/1023, tf.ST2084)

    return luminance_list


def create_n_point_luminance_patch(num_of_sample: int):
    luminance_list = create_luminance_value_array(num_of_array=num_of_sample)
    for luminance in luminance_list:
        create_heif_luminance_patch(luminance=luminance)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_heif_luminance_patch(luminance=1000)
    # create_heif_luminance_patch(luminance=500)
    # create_heif_luminance_patch(luminance=100)

    create_n_point_luminance_patch(num_of_sample=33)
