# -*- coding: utf-8 -*-
"""
calculate parameters for the ICC Profile
========================================

"""

# import standard libraries
import os

# import third-party libraries
from colour.adaptation import chromatic_adaptation_matrix_VonKries
from colour import xy_to_XYZ, XYZ_to_xy

# import my libraries
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


PCS_D50_XYZ = [0.9642, 1.0, 0.8249]
PCS_D50 = XYZ_to_xy(PCS_D50_XYZ)
D65 = cs.D65


def calc_chromatic_adaptation_matrix(src_white=cs.D65, dst_white=cs.D50):
    """
    calculate the chromatic adaptation matrix.
    the method is 'bradford'.

    Parameters
    ----------
    src_white : ndarray
        source white point. (ex. [0.3127, 0.3290])
    dst_white : ndarray
        distination white point.

    Returns
    -------
    ndarray
        A 3x3 matrix.

    Examples
    --------
    >>> calc_chromatic_adaptation_matrix(src_white=cs.D65, dst_white=cs.D50)
    [[ 1.04257389  0.03089108 -0.05281257]
     [ 0.02219345  1.00185663 -0.02107375]
     [-0.00116488 -0.00342053  0.76178908]]
    """
    src_XYZ = xy_to_XYZ(src_white)
    dst_XYZ = xy_to_XYZ(dst_white)
    mtx = chromatic_adaptation_matrix_VonKries(
        src_XYZ, dst_XYZ, 'Bradford')

    return mtx


def calc_rgb_to_xyz_mtx_included_chad_mtx(
        rgb_primaries, src_white, dst_white):
    """
    calculate the RGB to XYZ Matrix including the
    chromatic adaptation matrix.

    Parameters
    -----------
    rgb_primaries : ndarray
        prinmaries like [[ 0.708  0.292], [ 0.17   0.797], [ 0.131  0.046]].
    src_white : ndarray
        source white point. (ex. [0.3127, 0.3290])
    dst_white : ndarray
        distination white point.

    Returns
    -------
    ndarray
        A 3x3 matrix.

    Examples
    --------
    >>> calc_rgb_to_xyz_mtx_included_chad_mtx(
    ...     rgb_primaries=cs.get_primaries(cs.P3_D65),
    ...     src_white=cs.D65, dst_white=cs.D50)
    [[ 0.51514644  0.29200998  0.15713925]
     [ 0.24120032  0.69222254  0.06657714]
     [-0.00105014  0.04187827  0.78427647]]
    """
    src_white_XYZ = xy_to_XYZ(src_white)
    rgb_to_xyz_mtx = cs.calc_rgb_to_xyz_matrix(
        gamut_xy=rgb_primaries, white_large_xyz=src_white_XYZ)

    chad_mtx = calc_chromatic_adaptation_matrix(
        src_white=src_white, dst_white=dst_white)

    output_mtx = chad_mtx.dot(rgb_to_xyz_mtx)

    return output_mtx


def main_func():
    calc_chromatic_adaptation_matrix()
    calc_rgb_to_xyz_mtx_included_chad_mtx(
        rgb_primaries=cs.get_primaries(cs.P3_D65),
        src_white=cs.D65, dst_white=cs.D50)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # main_func()
    from colour import XYZ_to_xy
    import numpy as np
    large_xyz = np.array([0.96420288, 1.00000000, 0.82490540])
    xy = XYZ_to_xy(large_xyz)
    print(xy)
    print(cs.D50)
