# -*- coding: utf-8 -*-
"""
"""

# import standard libraries
import os
from turtle import heading

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


def scroll_image(img, offset_h, offset_v):
    """
    Scroll the input image based on the offset parameters.

    Parameters
    ----------
    img : ndarray
        A image data.
    offset_h : int
        A horizontal offset
    offset_v : int
        A vertical offset

    Returns
    -------
    ndarray
        A scrolled image data
    """
    height, width = img.shape[:2]
    out_img = np.zeros_like(img)
    pos_h = offset_h % width
    pos_v = offset_v % height

    pt1_h = 0
    pt1_v = 0
    pt2_h = pos_h
    pt2_v = 0
    pt3_h = pos_h
    pt3_v = pos_v
    pt4_h = width
    pt4_v = pos_v
    pt5_h = 0
    pt5_v = pos_v
    pt6_h = pos_h
    pt6_v = height
    pt7_h = width
    pt7_v = height

    out_img[:height-pos_v, :width-pos_h] = img[pt3_v:pt7_v, pt3_h:pt7_h]
    out_img[:height-pos_v, width-pos_h:] = img[pt5_v:pt6_v, pt5_h:pt6_h]
    out_img[height-pos_v:, :width-pos_h] = img[pt2_v:pt4_v, pt2_h:pt4_h]
    out_img[height-pos_v:, width-pos_h:] = img[pt1_v:pt3_v, pt1_h:pt3_h]

    return out_img


def debug_func_scroll_image():
    base_img = tpg.img_read("./img/scale_tp_1920x1080.png")
    height, width = base_img.shape[:2]
    num_of_step = 8
    step_h = -width // num_of_step
    step_v = -height // num_of_step
    for idx in range(num_of_step * 2):
        offset_h = step_h * idx
        offset_v = step_v * idx
        out_img = scroll_image(
            img=base_img, offset_h=offset_h, offset_v=offset_v)
        fname = f"./debug/scale_{idx:08d}.png"
        print(fname)
        tpg.img_write(fname, out_img)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_func_scroll_image()
