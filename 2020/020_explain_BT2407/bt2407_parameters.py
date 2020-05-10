# -*- coding: utf-8 -*-
"""
パラメータ置き場
================

"""

# import standard libraries
import os

# import third-party libraries

# import my libraries
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


L_SAMPLE_NUM_MAX = 1024
H_SAMPLE_NUM_MAX = 1024

GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE = 1024
GAMUT_BOUNDARY_LUT_HUE_SAMPLE = 1024


def get_gamut_boundary_lut_name(
        color_space_name=cs.BT709,
        luminance_sample_num=GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE,
        hue_sample_num=GAMUT_BOUNDARY_LUT_HUE_SAMPLE):
    name = f"./luts/GamutBoundaryLUT_{color_space_name}_"\
        + f"L_{luminance_sample_num}_H_{hue_sample_num}.npy"
    return name


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(get_gamut_boundary_lut_name(cs.BT709, 10, 20))
