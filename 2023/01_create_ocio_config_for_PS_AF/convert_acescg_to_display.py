# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import RGB_to_RGB
from colour.models import RGB_COLOURSPACE_ACESCG, RGB_COLOURSPACE_BT709,\
    RGB_COLOURSPACE_BT2020

# import my libraries
import test_pattern_generator2 as tpg
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    linear_img_fname = "./img/video_monitor_input.png"
    linear_img = tpg.img_read_as_float(linear_img_fname)
    linear_img = (linear_img - (16/255)) * (255/(235-16))
    bt2020_linear_img = RGB_to_RGB(
        linear_img, RGB_COLOURSPACE_ACESCG, RGB_COLOURSPACE_BT2020)
    hdr10_img = tf.oetf(np.clip(bt2020_linear_img/100, 0, 1), tf.ST2084)
    hdr10_fname = "./img/video_monitor_2020_pq.png"
    print(hdr10_fname)
    tpg.img_wirte_float_as_16bit_int(hdr10_fname, hdr10_img)
    tpg.img_wirte_float_as_16bit_int(
        "./img/video_monitor_input_range_ex.png", linear_img)
