# -*- coding: utf-8 -*-
"""
create hue-chroma pattern
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np

# import my libraries
import color_space as cs
from create_gamut_booundary_lut\
    import calc_chroma_boundary_specific_ligheness_jzazbz
from common import MeasureExecTime

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []




if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

