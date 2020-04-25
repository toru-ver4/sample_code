# -*- coding: utf-8 -*-
"""
3DLUTの適用
===========

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import read_LUT, write_image, read_image

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def apply_hdr10_to_turbo_3dlut(src_img_name, dst_img_name, lut_3d_name):
    """
    HDR10の静止画に3DLUTを適用して Turbo の輝度マップを作る。
    """
    hdr_img = read_image(src_img_name)
    lut3d = read_LUT(lut_3d_name)
    luminance_map_img = lut3d.apply(hdr_img)
    write_image(luminance_map_img, dst_img_name, bit_depth='uint16')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    apply_hdr10_to_turbo_3dlut(
        src_img_name="./figure/step_ramp.tiff",
        dst_img_name="./figure/3dlut_sample_turbo.png",
        lut_3d_name="./3dlut/PQ_BT2020_to_Turbo_sRGB.cube")
