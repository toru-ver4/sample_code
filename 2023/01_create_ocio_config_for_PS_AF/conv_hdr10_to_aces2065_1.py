# -*- coding: utf-8 -*-
"""
debug code
==========

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour.io import read_image, write_image
from colour.models import eotf_inverse_ST2084, eotf_ST2084,\
    RGB_COLOURSPACE_ACES2065_1, RGB_COLOURSPACE_BT2020, RGB_COLOURSPACE_BT709
from colour import RGB_to_RGB

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def main_func():
    src_fname\
        = "./img/SMPTE ST2084_ITU-R BT.2020_D65_1920x1080_rev05_type1.png"
    dst_fname = "./img/ST2084_BT.2020_D65_to_ACES2605-1_1920x1080.exr"

    nominal_to_luminance_ratio = 100

    hdr10_st2084 = read_image(path=src_fname)
    hdr10_linear = eotf_ST2084(hdr10_st2084) / nominal_to_luminance_ratio
    aces_2065_1 = RGB_to_RGB(
        RGB=hdr10_linear, input_colourspace=RGB_COLOURSPACE_BT2020,
        output_colourspace=RGB_COLOURSPACE_ACES2065_1,
        chromatic_adaptation_transform='CAT02')

    write_image(image=aces_2065_1, path=dst_fname)


def debug_chad_mtx():
    import color_space as cs
    from colour.adaptation import matrix_chromatic_adaptation_VonKries
    from colour import xy_to_XYZ, matrix_RGB_to_RGB
    from scipy import linalg
    src_fname\
        = "./img/SMPTE ST2084_ITU-R BT.2020_D65_1920x1080_rev05_type1.png"
    img = read_image(src_fname)
    img_linear = eotf_ST2084(img) / 100

    d65_xyz = cs.D65_XYZ
    d60_xyz = cs.D60_ACES_XYZ

    aces_primaries = cs.get_primaries(cs.ACES_AP0)
    bt2020_primaries = cs.get_primaries(cs.BT2020)
    rgb_to_xyz_mtx = cs.calc_rgb_to_xyz_matrix(bt2020_primaries, d65_xyz)
    rgb_to_xyz_mtx_temp = cs.calc_rgb_to_xyz_matrix(aces_primaries, d65_xyz)
    xyz_to_rgb_mtx = linalg.inv(rgb_to_xyz_mtx_temp)
    chad_mtx = matrix_chromatic_adaptation_VonKries(d65_xyz, d60_xyz)

    rgb_to_rgb_mtx = xyz_to_rgb_mtx.dot(chad_mtx.dot(rgb_to_xyz_mtx))
    print(rgb_to_rgb_mtx)

    mtx = matrix_RGB_to_RGB(
        RGB_COLOURSPACE_BT2020, RGB_COLOURSPACE_ACES2065_1, )
    print(mtx)
    # fname = "./img/ST2084_BT.2020_D65_to_ACES2605-1_1920x1080.exr"
    # img = read_image(fname)

    # data = img[820, 1858]
    # print(data)


def debug_linear_rgb_value():
    linear = 0.18
    import transfer_functions as tf
    non_linear = np.round(tf.oetf(linear, tf.GAMMA24) * 255)
    print(non_linear)
    print(48/255)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # debug_chad_mtx()
    debug_linear_rgb_value()
    # main_func()
