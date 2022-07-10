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
from colour.algebra import vector_dot

# import my libraries
import test_pattern_generator2 as tpg
import color_space as cs
from spectrum import DisplaySpectrum, create_display_sd,\
    CIE1931_CMFS, CIE2012_CMFS, ILLUMINANT_E

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
    XYZ = sd_to_XYZ(sd=color_checker_sds, cmfs=cmfs, illuminant=illuminant)
    # xyY = XYZ_to_xyY(XYZ)

    return XYZ


def load_color_checker_sr():
    # prepaere_color_checker_sr_data()
    large_xyz = sample_calc_sd_to_XYZ(
        spectral_shape=SpectralShape(380, 730, 1))
    rgb = cs.large_xyz_to_rgb(
        xyz=large_xyz, color_space_name=cs.BT709,
        xyz_white=cs.D65, rgb_white=cs.D65)
    rgb = np.clip(rgb/100, 0.0, 1.0) ** (1/2.4)
    img = tpg.plot_color_checker_image(rgb=rgb)
    tpg.img_wirte_float_as_16bit_int("./figure/color_checker.png", img)


def numpy_mult_check():
    r_coef = np.arange(6) + 1
    r_coef = r_coef.reshape(1, 6)
    sd = np.linspace(0, 1, 8).reshape(8, 1)
    print(r_coef)
    print(sd)
    yy = r_coef * sd
    print(yy)


def debug_calc_msd_from_rgb_gain(
        r_mu=620, r_sigma=12, g_mu=535, g_sigma=18, b_mu=458, b_sigma=8,
        fname="./figure/sample222.png"):
    msd = create_display_sd(
        r_mu=r_mu, r_sigma=r_sigma,
        g_mu=g_mu, g_sigma=g_sigma,
        b_mu=b_mu, b_sigma=b_sigma,
        normalize_y=True)
    spectral_shape = SpectralShape(360, 780, 1)
    ds = DisplaySpectrum(msd=msd)
    cc_large_xyz = sample_calc_sd_to_XYZ(spectral_shape=spectral_shape)
    xyz_to_rgb_mtx = ds.get_rgb_to_xyz_mtx()

    rgb_gain = vector_dot(xyz_to_rgb_mtx, cc_large_xyz) / 100
    rgb_gain = np.clip(rgb_gain, 0.0, 1.0)

    cc_spectrum = ds.calc_msd_from_rgb_gain(rgb=rgb_gain)

    cmfs_1931 = CIE1931_CMFS
    cmfs_2012 = CIE2012_CMFS
    illuminant = ILLUMINANT_E

    cmfs_1931 = cmfs_1931.trim(shape=spectral_shape)
    cmfs_2012 = cmfs_2012.trim(shape=spectral_shape)
    cc_spectrum = cc_spectrum.interpolate(shape=spectral_shape)
    illuminant = illuminant.interpolate(shape=spectral_shape)

    xyz_1931 = sd_to_XYZ(
        sd=cc_spectrum, cmfs=cmfs_1931, illuminant=illuminant)
    xyz_2012 = sd_to_XYZ(
        sd=cc_spectrum, cmfs=cmfs_2012, illuminant=illuminant)

    rgb_1931 = cs.large_xyz_to_rgb(
        xyz=xyz_1931, color_space_name=cs.BT709)
    rgb_2012 = cs.large_xyz_to_rgb(
        xyz=xyz_2012, color_space_name=cs.BT709)

    rgb_1931 = np.clip(rgb_1931 / 100, 0.0, 1.0)
    rgb_2012 = np.clip(rgb_2012 / 100, 0.0, 1.0)

    img_cat = tpg.plot_color_checker_image(
        rgb=rgb_1931, rgb2=rgb_2012, size=(1280, 720))
    img_1931 = tpg.plot_color_checker_image(
        rgb=rgb_1931, rgb2=None, size=(1280, 720))
    img_2012 = tpg.plot_color_checker_image(
        rgb=rgb_2012, rgb2=None, size=(1280, 720))

    tpg.img_wirte_float_as_16bit_int(fname, img_cat**(1/2.4))
    tpg.img_wirte_float_as_16bit_int(fname+"_1931.png", img_1931**(1/2.4))
    tpg.img_wirte_float_as_16bit_int(fname+"_2012.png", img_2012**(1/2.4))

    # print(rgb_gain)


def debug_func():
    numpy_mult_check()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # load_color_checker_sr()
    # debug_func()
    debug_calc_msd_from_rgb_gain(
        r_mu=649, r_sigma=35, g_mu=539, g_sigma=33, b_mu=460, b_sigma=13,
        fname="./figure/metamerism_cc_bt709.png")
    debug_calc_msd_from_rgb_gain(
        r_mu=620, r_sigma=12, g_mu=535, g_sigma=18, b_mu=458, b_sigma=8,
        fname="./figure/metamerism_cc_p3.png")
    debug_calc_msd_from_rgb_gain(
        r_mu=639, r_sigma=3, g_mu=530, g_sigma=4, b_mu=465, b_sigma=4,
        fname="./figure/metamerism_cc_bt2020.png")
