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


def create_10bit_ramp_tp():
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


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_10bit_ramp_tp()
