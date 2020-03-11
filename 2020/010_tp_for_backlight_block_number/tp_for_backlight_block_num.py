# -*- coding: utf-8 -*-
"""
Make a test pattern video to check the number of the backlight block.
==============

Description.

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
import cv2

# import my libraries
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def draw_rectangle(img, square_obj, size, st_pos_h, st_pos_v):
    """
    Parameters
    ----------
    img : array_like (np.uint8)
        background_image
    size : int
        the size of the bright square object.
    st_pos_h : int
        start position of the square object that moves horizontally.
    st_pos_v : int
        start position of the square object that moves vertically.

    Returns
    -------
    img : array_like (np.int8)
        image with two rectangle.
    """
    tpg.merge(img, square_obj, (st_pos_h, 0))  # H方向に動く正方形
    tpg.merge(img, square_obj, (0, st_pos_v))  # V方向に動く正方形

    return img


def save_image(img, width, height, size, frame_rate, frame_idx):
    fname = f"./sequence/tp_{width}x{height}_{frame_rate}p_{frame_idx:04d}.png"
    cv2.imwrite(fname, img)


def constant_velocity_linear_motion(
        width=1920, height=1080, size=64, frame_rate=60, second=10):
    """
    Parameters
    ----------
    width : int
        the width of the test pattern video
    height : int
        the height of the test pattern video
    size : int
        the size of the bright square object.
    frame_rate : int
        framerate of the test pattern video
    second : int
        time length of the test pattern video

    Returns
    -------
    None
        png sequence files are saved to "sequence" directory.
    """
    frame_num = frame_rate * second
    h_distance_total = width - size
    v_distance_total = height - size

    # "-1" の計算は始点に関しては計算の必要が無いため
    h_distance_list = tpg.equal_devision(h_distance_total, frame_num - 1)
    v_distance_list = tpg.equal_devision(v_distance_total, frame_num - 1)

    base_img = np.zeros((height, width, 3), dtype=np.uint8)
    square_obj = np.ones((size, size, 3), dtype=np.uint8) * 192

    # 最初の画像はループ外でプロット
    temp_img = draw_rectangle(
        img=base_img.copy(), square_obj=square_obj,
        size=size, st_pos_h=0, st_pos_v=0)
    save_image(temp_img, width, height, size, frame_rate, frame_idx=0)

    for idx in range(1, frame_num):
        st_pos_h = np.sum(h_distance_list[:idx])
        st_pos_v = np.sum(v_distance_list[:idx])
        temp_img = draw_rectangle(
            img=base_img.copy(), square_obj=square_obj, size=size,
            st_pos_h=st_pos_h, st_pos_v=st_pos_v)
        save_image(
            img=temp_img, width=width, height=height, size=size,
            frame_rate=frame_rate, frame_idx=idx)


def main_func():
    width = 1920
    height = 1080
    fps = 60
    sec = 10
    size = width // 32
    constant_velocity_linear_motion(
        width=width, height=height, size=size, frame_rate=fps, second=sec)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
