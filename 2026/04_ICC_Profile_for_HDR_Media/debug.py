# -*- coding: utf-8 -*-
"""
debug code
==========

"""

# import standard libraries
import os
import sys

# import third-party libraries
import numpy as np

# import my libraries
from test_pattern_generator2 import add_icc_profile_using_exiftool

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []




if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_gamma24_bt2020()
    # create_mhc2_profile_with_gain(
    #     gain=0.5,
    #     min_luminance=0.1,
    #     peak_luminance=700,
    #     max_full_frame_luminance=700,
    #     cs_name=cs.BT2020
    # )
    create_bt2020_pq_curve_4096_with_cicp_profile(cicp=[9, 16, 0, 1])
