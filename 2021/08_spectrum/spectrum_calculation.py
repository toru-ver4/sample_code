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
from colour import XYZ_to_xyY, XYZ_to_RGB, SpragueInterpolator,\
    SpectralDistribution
from colour.temperature import CCT_to_xy_CIE_D
from colour import sd_CIE_illuminant_D_series, SpectralShape
from colour.colorimetry import MSDS_CMFS_STANDARD_OBSERVER
from colour.utilities import tstack
from colour.models import RGB_COLOURSPACE_BT709

# import my libraries
from test_pattern_generator2 import D65_WHITE


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


VALID_WAVELENGTH_ST = 360
VALID_WAVELENGTH_ED = 830
VALID_SHAPE = SpectralShape(
    VALID_WAVELENGTH_ST, VALID_WAVELENGTH_ED, 1)


REFRECT_100P_SD = SpectralDistribution(
    data=dict(
        zip(np.arange(VALID_WAVELENGTH_ST, VALID_WAVELENGTH_ED+1),
            np.ones(VALID_WAVELENGTH_ED-VALID_WAVELENGTH_ST+1))))


def calc_illuminant_d_spectrum(color_temp=6500):
    xy = CCT_to_xy_CIE_D(color_temp)
    print(xy)
    sd = sd_CIE_illuminant_D_series(xy)
    sd.values = sd.values / 100

    return sd.interpolate(
        shape=VALID_SHAPE, interpolator=SpragueInterpolator).trim(VALID_SHAPE)


def get_cie_2_1931_cmf():
    return MSDS_CMFS_STANDARD_OBSERVER['cie_2_1931']


def calc_xyY_from_single_spectrum(src_sd, ref_sd, cmfs):
    """
    Parameters
    ----------
    src_sd : SpectralDistribution
        light source
    ref_sd : SpectralDistribution
        refrectance
    cmfs : MultiSpectralDistributions
        cmfs
    """
    return XYZ_to_xyY(calc_xyz_from_single_spectrum(
        src_sd=src_sd, ref_sd=ref_sd, cmfs=cmfs))


def calc_linear_rgb_from_single_spectrum(src_sd, ref_sd, cmfs, color_space):
    """
    Parameters
    ----------
    src_sd : SpectralDistribution
        light source
    ref_sd : SpectralDistribution
        refrectance
    cmfs : MultiSpectralDistributions
        cmfs
    """
    large_xyz = calc_xyz_from_single_spectrum(
        src_sd=src_sd, ref_sd=ref_sd, cmfs=cmfs)
    linear_rgb = XYZ_to_RGB(
        large_xyz, D65_WHITE, D65_WHITE, color_space.matrix_XYZ_to_RGB)

    normalize_xyz = calc_xyz_from_single_spectrum(
        src_sd=src_sd, ref_sd=REFRECT_100P_SD, cmfs=cmfs)
    normalize_rgb = XYZ_to_RGB(
        normalize_xyz, D65_WHITE, D65_WHITE, color_space.matrix_XYZ_to_RGB)

    return linear_rgb / np.max(normalize_rgb)


def calc_xyz_from_single_spectrum(src_sd, ref_sd, cmfs):
    """
    Parameters
    ----------
    src_sd : SpectralDistribution
        light source
    ref_sd : SpectralDistribution
        refrectance
    cmfs : MultiSpectralDistributions
        cmfs
    """
    large_x = np.sum((src_sd.values * ref_sd.values) * cmfs.values[..., 0])
    large_y = np.sum((src_sd.values * ref_sd.values) * cmfs.values[..., 1])
    large_z = np.sum((src_sd.values * ref_sd.values) * cmfs.values[..., 2])

    normalize_coef = np.sum(src_sd.values * cmfs.values[..., 1])

    large_xyz = tstack([large_x, large_y, large_z]) / normalize_coef

    return large_xyz


def debug_func():
    color_temp = 10000
    estimated_xy = CCT_to_xy_CIE_D(color_temp)
    src_sd = calc_illuminant_d_spectrum(color_temp)
    ref_sd = REFRECT_100P_SD
    cmfs = get_cie_2_1931_cmf()
    result_xyY = calc_xyY_from_single_spectrum(
        src_sd=src_sd, ref_sd=ref_sd, cmfs=cmfs)

    result_rgb = calc_linear_rgb_from_single_spectrum(
        src_sd=src_sd, ref_sd=ref_sd, cmfs=cmfs,
        color_space=RGB_COLOURSPACE_BT709)
    print(f"estimated={estimated_xy}, result={result_xyY}")
    print(f"rgb_linear={result_rgb}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_func()
