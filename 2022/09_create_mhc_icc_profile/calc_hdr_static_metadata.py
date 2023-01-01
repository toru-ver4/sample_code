# -*- coding: utf-8 -*-
"""

==========

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def calc_max_luminance_code_value(luminance=1000):
    cv = 32 * np.log2(luminance/50)
    cv = np.uint8(np.round(np.clip(cv, 0, 255)))

    return cv


def calc_min_luminance_code_value(
        max_luminance=1000, min_luminance=0.1):
    cv = 255 * ((100 * min_luminance/max_luminance) ** 0.5)
    cv = np.uint8(np.round(np.clip(cv, 0, 255)))

    return cv


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cv = calc_max_luminance_code_value(luminance=1000)
    print(cv)
    cv = calc_max_luminance_code_value(luminance=10000)
    print(cv)

    cv = calc_min_luminance_code_value(
        max_luminance=1000, min_luminance=0.1)
    print(cv)
    cv = calc_min_luminance_code_value(
        max_luminance=10000, min_luminance=0.0005)
    print(cv)
