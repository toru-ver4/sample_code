# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import normalised_primary_matrix

# import my libraries
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2023 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    matrix_bt709 = normalised_primary_matrix(
        cs.get_primaries(cs.BT709), cs.D65)
    print(matrix_bt709)

    matrix_p3d65 = normalised_primary_matrix(
        cs.get_primaries(cs.P3_D65), cs.D65)
    print(matrix_p3d65)

    matrix_bt2020 = normalised_primary_matrix(
        cs.get_primaries(cs.BT2020), cs.D65)
    print(matrix_bt2020)
