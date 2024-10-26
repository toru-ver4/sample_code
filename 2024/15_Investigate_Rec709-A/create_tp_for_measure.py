# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour.io import ImageAttribute_Specification, write_image

# import my libraries
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def create_10bit_ramp_for_eotf():
    width = 1280
    height = 720
    num_of_sample = 1024
    line = np.zeros(width, dtype=np.uint16)
    line[:num_of_sample] = np.arange(num_of_sample)
    line = line / 1023
    img = tpg.h_mono_line_to_img(line=line, height=height)

    bit_option = ImageAttribute_Specification("oiio:BitsPerSample", 10)
    write_image(
        img, path="./img/10-bit_ramp.dpx", bit_depth='uint16',
        attributes=[bit_option]
    )
    tpg.img_wirte_float_as_16bit_int("./src_img/10-bit_ramp.png", img)


def create_log2_ramp_for_oetf():
    width = 1280
    height = 720

    target_black_val = 0.00003
    target_white_val = 1.0

    ref_val = 0.18
    max_exp = np.log2(target_white_val / ref_val)
    min_exp = np.log2(target_black_val / ref_val)
    x = tpg.get_log2_x_scale(
        sample_num=width, ref_val=ref_val,
        min_exposure=min_exp, max_exposure=max_exp)
    img = tpg.h_mono_line_to_img(x, height)
    fname = f"./src_img/src_log2_{min_exp:.3f}_to_{max_exp:.3f}_stops.exr"
    print(fname)
    write_image(img, fname)


def create_linear_ramp_tp():
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_10bit_ramp_for_eotf()
    create_log2_ramp_for_oetf()
