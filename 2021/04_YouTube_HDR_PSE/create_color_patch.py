# -*- coding: utf-8 -*-
"""
decode
======
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np

# import my libraries
import test_pattern_generator2 as tpg


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def make_plane_rgbmyc_color_image():
    color_list = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1],
         [1, 0, 1], [1, 1, 0], [0, 1, 1]])
    color_name_list = ['r', 'g', 'b', 'm', 'y', 'c']

    for color, color_name in zip(color_list, color_name_list):
        img = np.ones((1080, 1920, 3)) * (color * 1023 / 1023)
        fname = f"./img/plane_color_{color_name}.png"
        tpg.img_wirte_float_as_16bit_int(fname, img)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    make_plane_rgbmyc_color_image()
