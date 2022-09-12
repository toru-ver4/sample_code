# -*- coding: utf-8 -*-
"""
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from scipy import linalg
from colour import sd_to_XYZ, SpectralShape
from colour.algebra import vector_dot
from colour.models import eotf_inverse_sRGB
from colour.io import write_image

# import my libraries
from spectrum import MultiSignals, MultiSpectralDistributions,\
    calc_rgb_to_xyz_matrix_from_spectral_distribution,\
    trim_and_iterpolate, MSDS_CMFS, SDS_ILLUMINANTS

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def load_display_spectrum(fname):
    data = np.loadtxt(fname=fname, delimiter=",")
    sd = data[..., 1:]
    domain = np.uint16(data[..., 0])
    signals = MultiSignals(data=sd, domain=domain)
    spd = MultiSpectralDistributions(data=signals)

    return spd


def prepaere_color_checker_sr_data():
    color_checker_sr_fname = "./ref_data/color_checker_sr.txt"
    data = np.loadtxt(
        fname=color_checker_sr_fname, delimiter='\t', skiprows=1).T
    domain = np.arange(380, 740, 10)
    color_checker_signals = MultiSignals(data=data, domain=domain)
    color_checker_sds = MultiSpectralDistributions(data=color_checker_signals)

    return color_checker_sds


def color_checker_calc_sd_to_XYZ_D65_illuminant():
    spectral_shape = SpectralShape(380, 730, 1)
    cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    illuminant = SDS_ILLUMINANTS['D65']
    color_checker_sds = prepaere_color_checker_sr_data()
    color_checker_sds = trim_and_iterpolate(color_checker_sds, spectral_shape)
    cmfs_trimed = trim_and_iterpolate(cmfs, spectral_shape)
    illuminant_intp = trim_and_iterpolate(illuminant, spectral_shape)
    XYZ = sd_to_XYZ(
        sd=color_checker_sds, cmfs=cmfs_trimed, illuminant=illuminant_intp)

    return XYZ


def main_func():
    np.set_printoptions(precision=4)

    # calc xyz_to_rgb_matrix
    display_spd = load_display_spectrum("./ref_data/ref_display_spd.csv")
    rgb_to_xyz_matrix = calc_rgb_to_xyz_matrix_from_spectral_distribution(
        spd=display_spd)
    xyz_to_rgb_matrix = linalg.inv(rgb_to_xyz_matrix)

    # calc Color Checker's XYZ
    maximum_large_y_d65 = 100
    color_checker_large_xyz = color_checker_calc_sd_to_XYZ_D65_illuminant()
    yellow_large_xyz = color_checker_large_xyz[15] / maximum_large_y_d65

    yellow_rgb = vector_dot(xyz_to_rgb_matrix, yellow_large_xyz)
    print(f"R_D, G_D, B_D = {yellow_rgb}")

    yellow_rgb_srgb = eotf_inverse_sRGB(yellow_rgb)
    yellow_rgb_srgb_8bit = np.uint8(np.round(yellow_rgb_srgb * 255))
    print(f"R'_D, G'_D, B'_D = {yellow_rgb_srgb_8bit}")

    img = np.ones((256, 256, 3), dtype=np.uint8) * yellow_rgb_srgb_8bit
    write_image(img/255, "./figure/yellow_sRGB.png", 'uint8')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
