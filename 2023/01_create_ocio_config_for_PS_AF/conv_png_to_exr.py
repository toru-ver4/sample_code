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

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def main_func():
    # src_fname\
    #     = "./img/SMPTE ST2084_ITU-R BT.2020_D65_1920x1080_rev05_type1.png"
    # dst_fname = "./img/BT.2020_ST2084_D65_1920x1080.exr"
    src_fname = "./img/P3D65_ST2084_1920x1080.png"
    dst_fname = "./img/P3D65_ST2084_1920x1080.exr"

    img = read_image(src_fname)
    write_image(img, dst_fname)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
