# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import XYZ_to_Oklab, xy_to_XYZ

# import my libraries
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2023 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def debug_plot_oklab():
    large_xyz = xy_to_XYZ(cs.D65)
    oklab = XYZ_to_Oklab(large_xyz)
    print(oklab)

    primary_rgb = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    primary_xyz = cs.rgb_to_large_xyz(primary_rgb, cs.BT2020)
    primary_oklab = XYZ_to_Oklab(primary_xyz)
    print(primary_oklab)
    primary_oklch = cs.oklab_to_oklch(primary_oklab)
    print(primary_oklch)
    primary_oklab_back = cs.oklch_to_oklab(primary_oklch)
    print(primary_oklab_back)

    oklab = cs.rgb_to_oklab(primary_rgb, cs.BT2020)
    print(oklab)
    print(cs.oklab_to_oklch(oklab))
    rgb = cs.oklab_to_rgb(oklab, cs.BT2020)
    print(rgb)


def check_primary_secondary_hue_angle():
    pp = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1],
         [1, 0, 1], [1, 1, 0], [0, 1, 1]])
    xyz = cs.rgb_to_large_xyz(pp, cs.BT709)
    oklab = cs.XYZ_to_Oklab(xyz)
    oklch = cs.oklab_to_oklch(oklab)
    print(oklch[..., 2])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # debug_plot_oklab()
    check_primary_secondary_hue_angle()
