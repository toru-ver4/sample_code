# -*- coding: utf-8 -*-
"""
spectrum
"""

# import standard libraries
import os
import sys

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from colour.temperature import CCT_to_xy_CIE_D
from colour import sd_CIE_illuminant_D_series
from colour.colorimetry import MSDS_CMFS_STANDARD_OBSERVER
# import my libraries


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def calc_illuminant_d_spectrum(color_temp=6500):
    xy = CCT_to_xy_CIE_D(color_temp)
    print(xy)
    sd = sd_CIE_illuminant_D_series(xy)
    sd.values = sd.values / 100

    return sd


def get_cie_2_1931_cmf():
    return MSDS_CMFS_STANDARD_OBSERVER['cie_2_1931']


def debug_illuminant_d():
    # calc_illuminant_d_spectrum(6500)
    cmf = get_cie_2_1931_cmf()
    # print(cmf.wavelengths)
    print(np.max(cmf.values))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_illuminant_d()
