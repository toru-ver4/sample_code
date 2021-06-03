# -*- coding: utf-8 -*-
"""
study Jzazbz color space
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import xy_to_XYZ, RGB_to_XYZ
from colour.models import RGB_COLOURSPACE_BT2020
from scipy.io import savemat, loadmat

# import my libraries
from jzazbz import large_xyz_to_jzazbz, jzazbz_to_large_xyz
import test_pattern_generator2 as tpg
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def create_xyz_values_for_test():
    color_checker_linear = tpg.generate_color_checker_rgb_value() * 100
    rgbmycw_linear_100 = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1],
         [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1]]) * 100
    rgbmycw_linear_10000 = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1],
         [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1]]) * 10000
    test_rgb_linear = np.concatenate(
        [color_checker_linear, rgbmycw_linear_100, rgbmycw_linear_10000])

    large_xyz = RGB_to_XYZ(
        test_rgb_linear, cs.D65, cs.D65,
        RGB_COLOURSPACE_BT2020.matrix_RGB_to_XYZ)

    return large_xyz


def convert_ndarray_to_matlab_array(ndarray):
    pass


def sample_jzazbz_conv():
    d65_xy = np.array([0.3127, 0.3290])
    large_xyz = xy_to_XYZ(d65_xy) * 100
    large_xyz = np.array([95.047, 100, 108.883])
    print(large_xyz)

    jab = large_xyz_to_jzazbz(xyz=large_xyz)
    print(jab)


def compare_reference_code():
    ref_jzazbz = loadmat("./result.mat")['result']

    large_xyz = create_xyz_values_for_test()
    my_jzazbz = large_xyz_to_jzazbz(xyz=large_xyz)

    np.testing.assert_array_almost_equal(ref_jzazbz, my_jzazbz, decimal=7)
    # print(my_jzazbz)


def check_inverse_function():
    large_xyz = create_xyz_values_for_test()
    jzazbz = large_xyz_to_jzazbz(large_xyz)
    inv_xyz = jzazbz_to_large_xyz(jzazbz)
    np.testing.assert_array_almost_equal(large_xyz, inv_xyz, decimal=7)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # sample_jzazbz_conv()
    # large_xyz = create_xyz_values_for_test()
    # save_data = dict(large_xyz=large_xyz)
    # savemat("./test_data.mat", save_data)
    compare_reference_code()
    check_inverse_function()
