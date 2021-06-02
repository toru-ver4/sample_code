# -*- coding: utf-8 -*-
"""
study Jzazbz color space
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour.utilities import tstack
from scipy import linalg

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def apply_matrix(mtx, aa, bb, cc):
    """
    Parameters
    ----------
    mts : ndarray
        3x3 matrix
    aa : ndarray
        1-D array
    bb : ndarray
        1-D array
    cc : ndarray
        1-D array

    Returns
    -------
    xx : ndarray
        1-D array
    yy : ndarray
        1-D array
    zz : ndarray
        1-D array
    """
    xx = mtx[0, 0] * aa + mtx[0, 1] * bb + mtx[0, 2] * cc
    yy = mtx[1, 0] * aa + mtx[1, 1] * bb + mtx[1, 2] * cc
    zz = mtx[2, 0] * aa + mtx[2, 1] * bb + mtx[2, 2] * cc

    return xx, yy, zz


def st2084_oetf_like(x):
    """
    Examples
    --------

    """
    n = 2610/(2**14)
    p = 1.7*2523/(2**5)
    # p = 2523/4096*128
    c1 = 3424/(2**12)
    c2 = 2413/(2**7)
    c3 = 2392/(2**7)

    y = ((c1 + c2*((x/10000)**n))/(1 + c3*((x/10000)**n)))**p

    return y


def st2084_eotf_like(x):
    """
    Examples
    --------

    """
    n = 2610/(2**14)
    p = 1.7*2523/(2**5)
    # p = 2523/4096*128
    c1 = 3424/(2**12)
    c2 = 2413/(2**7)
    c3 = 2392/(2**7)

    y = ((c1 - (x ** (1/p))) / (c3 * (x ** (1/p)) - c2)) ** (1/n)

    return y * 10000


def large_xyz_to_jzazbz(xyz):
    """
    convert from CIEXYZ to Jzazbz.

    Parameters
    ----------
    xyz : ndarray
        xyz data.

    Examples
    --------
    >>> large_xyz = np.array([95.047, 100, 108.883])
    >>> jab = large_xyz_to_jzazbz(xyz=large_xyz)
    [  1.67173549e-01  -1.34044078e-04  -8.24666380e-05]
    """
    # coefficients
    b = 1.15
    g = 0.66
    d = -0.56
    d0 = -1.6295499532821566e-11
    mtx_1 = np.array(
        [[0.41478972, 0.579999, 0.014648],
         [-0.20151, 1.120649, 0.0531008],
         [-0.0166008, 0.2648, 0.6684799]])
    mtx_2 = np.array(
        [[0.5, 0.5, 0],
         [3.524000, -4.066708, 0.542708],
         [0.199076, 1.096799, -1.295875]])

    # calc X' and Y'
    xx = xyz[..., 0]
    yy = xyz[..., 1]
    zz = xyz[..., 2]

    xx2 = b * xx - (b - 1) * zz
    yy2 = g * yy - (g - 1) * xx

    ll, mm, ss = apply_matrix(mtx_1, xx2, yy2, zz)

    lll = st2084_oetf_like(ll)
    mmm = st2084_oetf_like(mm)
    sss = st2084_oetf_like(ss)

    ii, aa, bb = apply_matrix(mtx_2, lll, mmm, sss)
    jj = ((1 + d) * ii) / (1 + d * ii) - d0

    jab = tstack([jj, aa, bb])

    return jab


def jzazbz_to_large_xyz(jzazbz):
    """
    convert from Jzazbz to CIEXYZ.

    Parameters
    ----------
    jzazbz : ndarray
        jzazbz data.

    Examples
    --------
    """
    b = 1.15
    g = 0.66
    d = -0.56
    d0 = -1.6295499532821566e-11

    jz = jzazbz[..., 0]
    az = jzazbz[..., 1]
    bz = jzazbz[..., 2]

    iz = (jz + d0) / (1 + d - d * (jz + d0))

    mtx_1 = np.array(
        [[0.41478972, 0.579999, 0.014648],
         [-0.20151, 1.120649, 0.0531008],
         [-0.0166008, 0.2648, 0.6684799]])
    mtx_1_inv = linalg.inv(mtx_1)
    mtx_2 = np.array(
        [[0.5, 0.5, 0],
         [3.524000, -4.066708, 0.542708],
         [0.199076, 1.096799, -1.295875]])
    mtx_2_inv = linalg.inv(mtx_2)

    lll, mmm, sss = apply_matrix(mtx_2_inv, iz, az, bz)
    ll = st2084_eotf_like(lll)
    mm = st2084_eotf_like(mmm)
    ss = st2084_eotf_like(sss)

    xx2, yy2, zz2 = apply_matrix(mtx_1_inv, ll, mm, ss)
    xx = (xx2 + (b - 1) * zz2) / b
    yy = (yy2 + (g - 1) * xx) / g
    zz = zz2

    return tstack([xx, yy, zz])


def calc_hue_from_ab(aa, bb):
    """
    calculate hue.
    output range is [0, 2pi).

    Examples
    --------
    >>> aa = np.array([1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0.0, 0.5, 0.99])
    >>> bb = np.array([0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5, -0.001])
    >>> hue = calc_hue_from_ab(aa, bb)
    [0.  45.  90.  135.  180.  225.  270.  315. 359.94212549]
    """
    hue = np.where(aa != 0, np.arctan(bb/aa), np.pi/2*np.sign(bb))
    add_pi_idx = (aa < 0) & (bb >= 0)
    sub_pi_idx = (aa < 0) & (bb < 0)
    hue[add_pi_idx] = hue[add_pi_idx] + np.pi
    hue[sub_pi_idx] = hue[sub_pi_idx] - np.pi

    hue[hue < 0] = hue[hue < 0] + 2 * np.pi

    return np.rad2deg(hue)


def jzazbz_to_jzczhz(jzazbz):
    """
    convert from Jzazbz to JzCzhz

    Parameters
    ----------
    jzazbz : ndarray
        jzazbz data.

    Examples
    --------
    >>> az = np.array([1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0.0, 0.5, 0.99])
    >>> bz = np.array([0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5, -0.001])
    >>> jz = np.linspace(0, 1, len(az))
    >>> jzazbz = tstack([jz, az, bz])
    >>> print(jzazbz)
    [[ 0.     1.     0.   ]
     [ 0.125  0.5    0.5  ]
     [ 0.25   0.     1.   ]
     [ 0.375 -0.5    0.5  ]
     [ 0.5   -1.     0.   ]
     [ 0.625 -0.5   -0.5  ]
     [ 0.75   0.    -1.   ]
     [ 0.875  0.5   -0.5  ]
     [ 1.     0.99  -0.001]]

    >>> jzczhz = jzazbz_to_jzczhz(jzazbz)
    >>> print(jzczhz)
    [[  0.00000000e+00   1.00000000e+00   0.00000000e+00]
     [  1.25000000e-01   7.07106781e-01   4.50000000e+01]
     [  2.50000000e-01   1.00000000e+00   9.00000000e+01]
     [  3.75000000e-01   7.07106781e-01   1.35000000e+02]
     [  5.00000000e-01   1.00000000e+00   1.80000000e+02]
     [  6.25000000e-01   7.07106781e-01   2.25000000e+02]
     [  7.50000000e-01   1.00000000e+00   2.70000000e+02]
     [  8.75000000e-01   7.07106781e-01   3.15000000e+02]
     [  1.00000000e+00   9.90000505e-01   3.59942125e+02]]
    """
    jz = jzazbz[..., 0]
    az = jzazbz[..., 1]
    bz = jzazbz[..., 2]

    cz = (az ** 2 + bz ** 2) ** 0.5
    hz = calc_hue_from_ab(az, bz)

    return tstack([jz, cz, hz])


def jzczhz_to_jzazbz(jzczhz):
    """
    convert from JzCzhz to Jzazbz

    Parameters
    ----------
    jzczhz : ndarray
        jzczhz data.

    Examples
    --------
    >>> jzczhz = np.array(
    ...     [[0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
    ...      [1.25000000e-01, 7.07106781e-01, 4.50000000e+01],
    ...      [2.50000000e-01, 1.00000000e+00, 9.00000000e+01],
    ...      [3.75000000e-01, 7.07106781e-01, 1.35000000e+02],
    ...      [5.00000000e-01, 1.00000000e+00, 1.80000000e+02],
    ...      [6.25000000e-01, 7.07106781e-01, 2.25000000e+02],
    ...      [7.50000000e-01, 1.00000000e+00, 2.70000000e+02],
    ...      [8.75000000e-01, 7.07106781e-01, 3.15000000e+02],
    ...      [1.00000000e+00, 9.90000505e-01, 3.59942125e+02]])
    >>> jzazbz_inverted = jzczhz_to_jzazbz(jzczhz)
    [[  0.00000000e+00   1.00000000e+00   0.00000000e+00]
     [  1.25000000e-01   5.00000000e-01   5.00000000e-01]
     [  2.50000000e-01   6.12323400e-17   1.00000000e+00]
     [  3.75000000e-01  -5.00000000e-01   5.00000000e-01]
     [  5.00000000e-01  -1.00000000e+00   1.22464680e-16]
     [  6.25000000e-01  -5.00000000e-01  -5.00000000e-01]
     [  7.50000000e-01  -1.83697020e-16  -1.00000000e+00]
     [  8.75000000e-01   5.00000000e-01  -5.00000000e-01]
     [  1.00000000e+00   9.90000000e-01  -1.00000855e-03]]
    """
    jz = jzczhz[..., 0]
    cz = jzczhz[..., 1]
    hz = jzczhz[..., 2]

    hz_rad = np.deg2rad(hz)
    az = cz * np.cos(hz_rad)
    bz = cz * np.sin(hz_rad)

    return tstack([jz, az, bz])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(st2084_eotf_like(0.4))
    print(st2084_oetf_like(100))
