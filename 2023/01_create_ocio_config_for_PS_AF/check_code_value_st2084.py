# -*- coding: utf-8 -*-
"""
debug code
==========

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np

# import my libraries
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def calc_10bit_st2084_cv(linear_val=0.18):
    cv = np.round(tf.oetf_from_luminance(linear_val*100, tf.ST2084) * 1023)
    return np.uint16(cv)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(calc_10bit_st2084_cv(0.08))
    print(calc_10bit_st2084_cv(0.10))
    print(calc_10bit_st2084_cv(0.18))
    print(calc_10bit_st2084_cv(1.0))
    print(calc_10bit_st2084_cv(10))
    print(calc_10bit_st2084_cv(100))
