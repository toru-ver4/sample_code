# -*- coding: utf-8 -*-
"""
debug code
==========

"""

# import standard libraries
import os

# import third-party libraries
from colour.models import RGB_COLOURSPACES
from colour import RGB_to_RGB

# import my libraries
import color_space as cs
import transfer_functions as tf
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def make_images(gamma_float=3.0):
    src_color_space = cs.ACES_AP0

    img = tpg.img_read_as_float(
        "./img/ColorChecker_All_ITU-R BT.709_D65_BT1886_Reverse.tiff")
    img_linear = img ** 2.4
    img_sRGB = tf.oetf(img_linear, tf.SRGB)
    ap1_img_linear = RGB_to_RGB(
        img_linear,
        RGB_COLOURSPACES[cs.BT709], RGB_COLOURSPACES[src_color_space])
    ap1_non_linear = ap1_img_linear ** (1/gamma_float)
    tpg.img_wirte_float_as_16bit_int("./img/ap0_img.png", ap1_non_linear)
    tpg.img_wirte_float_as_16bit_int("./img/sRGB.png", img_sRGB)


def main_func():
    make_images(gamma_float=3.0)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
