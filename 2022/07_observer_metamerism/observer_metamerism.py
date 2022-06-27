# -*- coding: utf-8 -*-
"""
"""

# import standard libraries
import os
import requests

# import third-party libraries
import numpy as np
from colour.continuous import MultiSignals
from colour import sd_to_XYZ, MultiSpectralDistributions, MSDS_CMFS,\
    SDS_ILLUMINANTS, SpectralShape

# import my libraries
import test_pattern_generator2 as tpg
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


DEFAULT_SPECTRAL_SHAPE = SpectralShape(380, 780, 1)


def download_file(url, color_checker_sr_fname):
    response = requests.get(url)
    open(color_checker_sr_fname, "wb").write(response.content)


def prepaere_color_checker_sr_data():
    """
    Returns
    -------
    MultiSpectralDistributions
        multi-spectral distributions data.
    """
    color_checker_sr_fname = "./ref_data/color_checker_sr.txt"
    data = np.loadtxt(
        fname=color_checker_sr_fname, delimiter='\t', skiprows=1).T
    domain = np.arange(380, 740, 10)
    color_checker_signals = MultiSignals(data=data, domain=domain)
    color_checker_sds = MultiSpectralDistributions(data=color_checker_signals)

    return color_checker_sds


def sample_calc_sd_to_XYZ(spectral_shape=DEFAULT_SPECTRAL_SHAPE):
    color_checker_sds = prepaere_color_checker_sr_data()
    color_checker_sds = color_checker_sds.interpolate(shape=spectral_shape)
    cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    cmfs = cmfs.trim(spectral_shape)
    illuminant = SDS_ILLUMINANTS['D65']
    illuminant = illuminant.interpolate(spectral_shape)
    # print(illuminant)
    print(color_checker_sds.values.shape, cmfs.values.shape, illuminant.values.shape)
    XYZ = sd_to_XYZ(sd=color_checker_sds, cmfs=cmfs, illuminant=illuminant)
    # xyY = XYZ_to_xyY(XYZ)

    return XYZ


def load_color_checker_sr():
    # prepaere_color_checker_sr_data()
    large_xyz = sample_calc_sd_to_XYZ(
        spectral_shape=SpectralShape(380, 730, 1))
    rgb = cs.large_xyz_to_rgb(
        xyz=large_xyz, color_space_name=cs.BT709,
        xyz_white=cs.D50, rgb_white=cs.D65)
    rgb = np.clip(rgb/100, 0.0, 1.0) ** (1/2.4)
    img = tpg.plot_color_checker_image(rgb=rgb)
    tpg.img_wirte_float_as_16bit_int("./figure/color_checker.png", img)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    load_color_checker_sr()
