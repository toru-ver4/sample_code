# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour.io import write_image, read_image

# import my libraries
import test_pattern_generator2 as tpg
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2023 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def oetf_st2084():
    i_name = "./img/Linear_BT.2020_1920x1080.exr"
    o_name = "./img/ST2084_BT.2020_1920x1080.exr"
    img = read_image(i_name)
    img = tf.oetf(img / 100, tf.ST2084)

    write_image(img, o_name)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    oetf_st2084()
