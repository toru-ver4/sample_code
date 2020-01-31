# -*- coding: utf-8 -*-
"""
輝点のパターンを作成する
=======================

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
import cv2

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def make_bright_point_img(
        bg_color=[0, 0, 0], fg_color=[192, 192, 192], point_size=4):
    width = 3840
    height = 2160
    fname = "./img/bright_point_size{:04d}_bg_level_{:04d}.tiff"
    fname = fname.format(point_size, bg_color[1])

    img = np.ones((height, width, 3), dtype=np.uint8)\
        * np.array(bg_color, dtype=np.uint8)
    st_pos = [(width // 2) - (point_size // 2),
              (height // 2) - (point_size // 2)]
    ed_pos = [st_pos[0] + point_size, st_pos[1] + point_size]
    rec = (st_pos[0], st_pos[1], point_size, point_size)
    cv2.rectangle(img=img, rec=rec, color=fg_color, thickness=-1)
    cv2.imwrite(fname, img[:, :, ::-1])


def main_func():
    bg_color_list = [[0, 0, 0], [64, 64, 64], [96, 96, 96]]
    fg_color = [192, 192, 192]
    point_size_list = [4, 16, 64, 256]

    for bg_color in bg_color_list:
        for point_size in point_size_list:
            make_bright_point_img(
                bg_color=bg_color, fg_color=fg_color, point_size=point_size)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
