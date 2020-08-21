# -*- coding: utf-8 -*-
"""
Tools for plot of the color volume
==================================

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import XYZ_to_RGB, xyY_to_XYZ, RGB_COLOURSPACES

# import my libraries
from color_volume_boundary_data import calc_xyY_boundary_data,\
    calc_xyY_boundary_data_log_scale
from common import MeasureExecTime
import color_space as cs
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def simple_linear_xyY_plot():
    xyY_obj = calc_xyY_boundary_data(
        color_space_name=cs.BT2020, y_num=512, h_num=1024)
    


def experimental_func():
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    experimental_func()
