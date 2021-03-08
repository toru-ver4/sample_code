# -*- coding: utf-8 -*-
"""
fix alpha blending of font_control
===================================
"""

# import standard libraries
import os
import cv2

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


def tiff2png():
    src = "./img/Gamma_2.4_ITU-R_BT.709_D65_1920x1080_rev04_type1.tiff"
    dst = "./img/Gamma_2.4_ITU-R_BT.709_D65_1920x1080_rev04_type1.png"
    img = tpg.img_read_as_float(src)
    tpg.img_wirte_float_as_16bit_int(dst, img)


def expand_4x(fname="hoge.png"):
    out_fname = fname + "_4x.png"
    img = tpg.img_read(fname)
    img_4x = cv2.resize(
        img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    tpg.img_write(out_fname, img_4x)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # tiff2png()
    expand_4x(fname="./img/YUV420.png")
    expand_4x(fname="./img/YUV444.png")
