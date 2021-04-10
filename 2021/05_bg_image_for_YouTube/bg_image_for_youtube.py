# -*- coding: utf-8 -*-
"""
create background image for YouTube Live Stream.
"""

# import standard libraries
import os
from colour.models.cie_lab import LCHab_to_Lab, Lab_to_XYZ
from colour.models.rgb.rgb_colourspace import XYZ_to_RGB

# import third-party libraries
import numpy as np
from colour import RGB_to_XYZ, XYZ_to_Lab, Lab_to_LCHab,\
    LCHab_to_Lab, Lab_to_XYZ, XYZ_to_RGB
from colour.models import BT709_COLOURSPACE
import cv2

# import my libraries
import test_pattern_generator2 as tpg
from test_pattern_coordinate import GridCoordinate
import transfer_functions as tf
from color_space import D65


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def create_bg_image(color=[0.5, 0.5, 0.5], width=1920, height=1080):
    img = np.ones((height, width, 3))
    img[..., :3] = img[..., :3] * color
    fname = "./img/bg_image.png"
    tpg.img_wirte_float_as_16bit_int(fname, img)


def create_fg_image(rgb=[201/255, 222/255, 251/255], width=1920, height=1080):
    st_chroma_rate = 1/3
    end_lch = sRGB_to_Lab(rgb=[201/255, 222/255, 251/255])
    end_chroma = end_lch[1]
    start_chroma = end_chroma * st_chroma_rate
    chroma_grad = np.linspace(start_chroma, end_chroma, height)
    chroma_grad = chroma_grad.reshape((height, 1, 1))
    chroma_grad_2d = chroma_grad * np.ones((1, width, 1))

    img = np.ones((height, width, 3)) * end_lch
    img[..., 1] = chroma_grad_2d[..., 0]

    large_xyz = Lab_to_XYZ(LCHab_to_Lab(img))
    rgb_linear = XYZ_to_RGB(
        large_xyz, D65, D65, BT709_COLOURSPACE.XYZ_to_RGB_matrix)
    img = tf.oetf(rgb_linear, tf.SRGB)

    # debug

    fname = "./img/fg_image.png"
    tpg.img_wirte_float_as_16bit_int(fname, img)


def sRGB_to_Lab(rgb=[201/255, 222/255, 251/255]):
    linear_rgb = tf.eotf(np.array(rgb), tf.SRGB)
    large_xyz = RGB_to_XYZ(
        linear_rgb, D65, D65, BT709_COLOURSPACE.RGB_to_XYZ_matrix)
    lch = Lab_to_LCHab(XYZ_to_Lab(large_xyz))

    return lch


def create_alpha_circle_image(
        width=1920, height=1080,
        h_num=3, v_num=2, fg_width=200, fg_height=200,
        min_rad=5, max_rad=50, gamma=1.0):
    gc = GridCoordinate(
        bg_width=width, bg_height=height,
        fg_width=fg_width, fg_height=fg_height,
        h_num=h_num, v_num=v_num, remove_tblr_margin=True)
    pos_list = gc.get_st_pos_list()
    rad_rate = (np.linspace(min_rad, max_rad, v_num) - min_rad)\
        / (max_rad - min_rad)
    rad_rate = rad_rate ** gamma
    rad_list = np.uint16(np.round(rad_rate * (max_rad - min_rad) + min_rad))
    print(rad_list)
    fg_img = np.ones((height, width, 3), dtype=np.uint8) * 255
    for v_idx in range(v_num):
        for h_idx in range(h_num):
            center_pos = pos_list[h_idx][v_idx]
            center_pos[0] = center_pos[0] + fg_width // 2
            center_pos[1] = center_pos[1] + fg_height // 2
            cv2.circle(
                fg_img, (center_pos[0], center_pos[1]), rad_list[v_idx],
                (0, 0, 0), -1, cv2.LINE_AA)
    tpg.img_wirte_float_as_16bit_int("./img/test_circle.png", fg_img/255)


def create_youtube_bg_image():
    bg_img = tpg.img_read_as_float("./img/bg_image.png")
    fg_img = tpg.img_read_as_float("./img/fg_image.png")
    alpha_img = tpg.img_read_as_float("./img/test_circle.png")[..., 1]
    alpha_img = alpha_img.reshape((bg_img.shape[0], bg_img.shape[1], 1))

    img = bg_img * alpha_img + fg_img * (1 - alpha_img)
    tpg.img_wirte_float_as_16bit_int("./img/youtube_background.png", img)


def main_func():
    create_bg_image(color=np.array([1, 1, 1])*245/255)
    create_fg_image(rgb=[174/255, 197/255, 244/255])
    create_alpha_circle_image(
        width=1920, height=1080,
        h_num=int(16*2), v_num=int(9*2), fg_width=2, fg_height=2,
        min_rad=5, max_rad=55, gamma=1.6)
    create_youtube_bg_image()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
