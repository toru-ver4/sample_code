# -*- coding: utf-8 -*-
"""
spectrum
"""

# import standard libraries
import os

import numpy as np
from scipy.stats import norm

from colour import XYZ_to_xyY, XYZ_to_RGB, xy_to_XYZ, SpragueInterpolator,\
    SpectralDistribution, MultiSpectralDistributions
from colour.temperature import CCT_to_xy_CIE_D
from colour import sd_CIE_illuminant_D_series, SpectralShape, Extrapolator
from colour.colorimetry import MSDS_CMFS_STANDARD_OBSERVER
from colour.utilities import tstack
from colour.models import RGB_COLOURSPACE_BT709
from colour.algebra import LinearInterpolator

# import my libraries
from test_pattern_generator2 import D65_WHITE, plot_color_checker_image,\
    img_wirte_float_as_16bit_int
import transfer_functions as tf
import plot_utility as pu
import matplotlib.pyplot as plt


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


VALID_WAVELENGTH_ST = 360
VALID_WAVELENGTH_ED = 830
# VALID_WAVELENGTH_ST = 380
# VALID_WAVELENGTH_ED = 730
VALID_SHAPE = SpectralShape(
    VALID_WAVELENGTH_ST, VALID_WAVELENGTH_ED, 1)


REFRECT_100P_SD = SpectralDistribution(
    data=dict(
        zip(np.arange(VALID_WAVELENGTH_ST, VALID_WAVELENGTH_ED+1),
            np.ones(VALID_WAVELENGTH_ED-VALID_WAVELENGTH_ST+1))))


def calc_illuminant_d_spectrum(color_temp=6500):
    xy = CCT_to_xy_CIE_D(color_temp)
    # print(xy)
    sd = sd_CIE_illuminant_D_series(xy)
    sd.values = sd.values / 100

    return sd.interpolate(
        shape=VALID_SHAPE, interpolator=SpragueInterpolator).trim(VALID_SHAPE)


def get_cie_2_1931_cmf():
    return MSDS_CMFS_STANDARD_OBSERVER['cie_2_1931'].trim(VALID_SHAPE)


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
    sd_result = src_sd.values * ref_sd.values
    large_x = np.sum(sd_result * cmfs.values[..., 0])
    large_y = np.sum(sd_result * cmfs.values[..., 1])
    large_z = np.sum(sd_result * cmfs.values[..., 2])

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

    result_rgb = calc_linear_rgb_from_spectrum(
        src_sd=src_sd, ref_sd=ref_sd, cmfs=cmfs,
        color_space=RGB_COLOURSPACE_BT709)
    print(f"estimated={estimated_xy}, result={result_xyY}")
    print(f"rgb_linear={result_rgb}")


def load_color_checker_spectrum():
    fname = "./data/babel_spectrum_2012.csv"
    data = np.loadtxt(fname, delimiter=',')
    wavelength = data[0]
    values = data[1:].T
    data = dict(zip(wavelength, values))
    color_checker_sd = MultiSpectralDistributions(data=data)
    color_checker_sd = color_checker_sd.interpolate(
        shape=VALID_SHAPE, interpolator=SpragueInterpolator)

    keyword = dict(
        method='Constant', left=0, right=0)
    color_checker_sd.extrapolate(
        shape=VALID_SHAPE, extrapolator=Extrapolator,
        extrapolator_kwargs=keyword)

    return color_checker_sd


def calc_xyz_from_multi_spectrum(src_sd, ref_sd, cmfs):
    """
    Parameters
    ----------
    src_sd : SpectralDistribution
        light source
    ref_sd : MultiSpectralDistributions
        refrectance
    cmfs : MultiSpectralDistributions
        cmfs
    """
    src_shape = src_sd.values.shape
    cmf_shape = cmfs.values.shape

    sd_result = src_sd.values.reshape((src_shape[0], 1)) * ref_sd.values

    large_x = np.sum(
        (sd_result) * cmfs.values[..., 0].reshape(cmf_shape[0], 1), 0)
    large_y = np.sum(
        (sd_result) * cmfs.values[..., 1].reshape(cmf_shape[0], 1), 0)
    large_z = np.sum(
        (sd_result) * cmfs.values[..., 2].reshape(cmf_shape[0], 1), 0)

    normalize_coef = np.sum(src_sd.values * cmfs.values[..., 1])

    large_xyz = tstack([large_x, large_y, large_z]) / normalize_coef

    return large_xyz


def calc_linear_rgb_from_spectrum(src_sd, ref_sd, cmfs, color_space):
    """
    Parameters
    ----------
    src_sd : SpectralDistribution
        light source
    ref_sd : SpectralDistribution or MultiSpectralDistributions
        refrectance
    cmfs : MultiSpectralDistributions
        cmfs
    """
    if isinstance(ref_sd, SpectralDistribution):
        calc_xyz_func = calc_xyz_from_single_spectrum
    elif isinstance(ref_sd, MultiSpectralDistributions):
        calc_xyz_func = calc_xyz_from_multi_spectrum
    else:
        print("Error: invalid 'ref_sd' type")
        calc_xyz_func = None
    large_xyz = calc_xyz_func(
        src_sd=src_sd, ref_sd=ref_sd, cmfs=cmfs)
    linear_rgb = XYZ_to_RGB(
        large_xyz, D65_WHITE, D65_WHITE, color_space.matrix_XYZ_to_RGB)
    # print(f"xyY={XYZ_to_xyY(large_xyz)}")

    normalize_xyz = calc_xyz_from_single_spectrum(
        src_sd=src_sd, ref_sd=REFRECT_100P_SD, cmfs=cmfs)
    normalize_rgb = XYZ_to_RGB(
        normalize_xyz, D65_WHITE, D65_WHITE, color_space.matrix_XYZ_to_RGB)

    return linear_rgb / np.max(normalize_rgb)


def calc_color_temp_after_spectrum_rendering(src_sd, cmfs):
    """
    Parameters
    ----------
    src_sd : SpectralDistribution
        light source
    cmfs : MultiSpectralDistributions
        cmfs
    """
    ref_sd = REFRECT_100P_SD
    result_xyY = calc_xyY_from_single_spectrum(
        src_sd=src_sd, ref_sd=ref_sd, cmfs=cmfs)

    return result_xyY[:2]


def get_color_checker_large_xyz_of_d65(color_temp):
    src_sd = calc_illuminant_d_spectrum(color_temp)
    ref_multi_sd = load_color_checker_spectrum()
    cmfs = get_cie_2_1931_cmf()
    large_xyz = calc_xyz_from_multi_spectrum(
        src_sd=src_sd, ref_sd=ref_multi_sd, cmfs=cmfs)

    return large_xyz


def convert_color_checker_linear_rgb_from_d65(
        d65_color_checker_xyz, dst_white, color_space):
    """
    convert from D65 white color checker to Dxx white color checker.

    Examples
    --------
    >>> d65_color_checker_xyz = get_color_checker_large_xyz_of_d65(6504)
    >>> convert_color_checker_linear_rgb_from_d65(
    ...     d65_color_checker_xyz=d65_color_checker_xyz,
    ...     dst_white=result_xy, color_space=RGB_COLOURSPACE_BT709)
    [[ 0.14683397  0.03958833  0.00627516]
     [ 0.47596101  0.14178862  0.02575557]
     [ 0.13636166  0.09700691  0.06000937]
     [ 0.11606916  0.07385164  0.00284443]
     [ 0.22484404  0.10575814  0.07761267]
     [ 0.23497984  0.25816364  0.05921924]
     [ 0.56708895  0.08951235 -0.01117377]
     [ 0.08001099  0.05211231  0.07652314]
     [ 0.41278372  0.03650183  0.01537575]
     [ 0.08808705  0.02052208  0.02608161]
     [ 0.39305027  0.24938059 -0.01730461]
     [ 0.65500321  0.16628094 -0.02019578]
     [ 0.03354099  0.02411967  0.05853887]
     [ 0.12997428  0.15023356 -0.00127581]
     [ 0.31587871  0.00976112  0.00245512]
     [ 0.7700667   0.2760175  -0.03421657]
     [ 0.38727265  0.03718964  0.05450171]
     [ 0.05200192  0.12495766  0.06890876]
     [ 0.91575358  0.44532093  0.12991015]
     [ 0.58456461  0.28768097  0.08856024]
     [ 0.35692614  0.17568433  0.05459322]
     [ 0.18895591  0.093656    0.02919013]
     [ 0.08791188  0.043792    0.01393372]
     [ 0.03185322  0.01554741  0.00501038]]
    """
    color_temp = 6504
    src_white = CCT_to_xy_CIE_D(color_temp)
    linear_rgb = XYZ_to_RGB(
        d65_color_checker_xyz, D65_WHITE, dst_white,
        color_space.matrix_XYZ_to_RGB)

    # normalize coefficient
    large_xyz = xy_to_XYZ(src_white)
    normalize_rgb = XYZ_to_RGB(
        large_xyz, D65_WHITE, dst_white, color_space.matrix_XYZ_to_RGB)
    print(f"normalize_rgb = {normalize_rgb}")

    return linear_rgb / np.max(normalize_rgb)


def color_checker_check_func():
    src_sd = calc_illuminant_d_spectrum(3000)
    ref_multi_sd = load_color_checker_spectrum()
    cmfs = get_cie_2_1931_cmf()
    linear_rgb = linear_rgb = calc_linear_rgb_from_spectrum(
        src_sd=src_sd, ref_sd=ref_multi_sd, cmfs=cmfs,
        color_space=RGB_COLOURSPACE_BT709)
    rgb_srgb = tf.oetf(np.clip(linear_rgb, 0.0, 1.0), tf.SRGB)
    # color_checker_img = plot_color_checker_image(
    #     rgb=rgb_srgb, size=(540, 360), block_size=1/4.5)
    # img_wirte_float_as_16bit_int("hoge.png", color_checker_img)
    result_xy = calc_color_temp_after_spectrum_rendering(
        src_sd=src_sd, cmfs=cmfs)
    d65_color_checker_xyz = get_color_checker_large_xyz_of_d65(6504)
    linear_rgb = convert_color_checker_linear_rgb_from_d65(
        d65_color_checker_xyz=d65_color_checker_xyz,
        dst_white=result_xy, color_space=RGB_COLOURSPACE_BT709)
    print(linear_rgb)
    rgb_srgb2 = tf.oetf(np.clip(linear_rgb, 0.0, 1.0), tf.SRGB)
    color_checker_img = plot_color_checker_image(
        rgb=rgb_srgb, rgb2=rgb_srgb2, size=(540, 360), block_size=1/4.5)
    img_wirte_float_as_16bit_int("hoge.png", color_checker_img)


def create_display_spectrum_test():
    x = np.arange()
    y = norm.pdf(x, 500, scale=30)
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Title",
        graph_title_size=None,
        xlabel="X Axis Label", ylabel="Y Axis Label",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None,
        return_figure=True)
    ax1.plot(x, y, label="aaa")
    plt.legend(loc='upper left')
    plt.show()
    plt.close(fig)


def extrapolator_test():
    ref_multi_sd = load_color_checker_spectrum()
    keyword = dict(
        method='Constant', left=0, right=0)
    ref_multi_sd_360_830 = ref_multi_sd.copy()
    ref_multi_sd_360_830.extrapolate(
        shape=VALID_SHAPE, extrapolator=Extrapolator,
        extrapolator_kwargs=keyword)

    x1 = ref_multi_sd.wavelengths
    y1 = ref_multi_sd.values[..., 18]

    x2 = ref_multi_sd_360_830.wavelengths
    y2 = ref_multi_sd_360_830.values[..., 18]

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Title",
        graph_title_size=None,
        xlabel="X Axis Label", ylabel="Y Axis Label",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None,
        return_figure=True)
    ax1.plot(x2, y2, label="with Extrapolation")
    ax1.plot(x1, y1, '--', label="original")
    plt.legend(loc='upper left')
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # debug_func()
    extrapolator_test()
