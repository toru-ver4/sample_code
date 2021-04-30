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

# import my libraries
from test_pattern_generator2 import D65_WHITE, plot_color_checker_image,\
    img_wirte_float_as_16bit_int, _get_cmfs_xy, get_primaries,\
    get_chromaticity_image
import transfer_functions as tf
import plot_utility as pu
import matplotlib.pyplot as plt
import color_space as cs


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


class DisplaySpectralDistribution():
    def __init__(
            self, wavelengths, r_mean, r_dist, r_gain,
            g_mean, g_dist, g_gain, b_mean, b_dist, b_gain):
        self.x = wavelengths
        self.update_spectrum(
            r_mean=r_mean, r_dist=r_dist, r_gain=r_gain,
            g_mean=g_mean, g_dist=g_dist, g_gain=g_gain,
            b_mean=b_mean, b_dist=b_dist, b_gain=b_gain)

    def create_norm(self, x, mean, dist, gain):
        return norm.pdf(x, loc=mean, scale=dist) * gain

    def update_spectrum(
            self, r_mean, r_dist, r_gain,
            g_mean, g_dist, g_gain, b_mean, b_dist, b_gain):
        self.r_values = self.create_norm(self.x, r_mean, r_dist, r_gain)
        self.g_values = self.create_norm(self.x, g_mean, g_dist, g_gain)
        self.b_values = self.create_norm(self.x, b_mean, b_dist, b_gain)
        self.w_values = self.r_values + self.g_values + self.b_values

        self.w_sd = SpectralDistribution(data=dict(zip(self.x, self.w_values)))
        self.r_sd = SpectralDistribution(data=dict(zip(self.x, self.r_values)))
        self.g_sd = SpectralDistribution(data=dict(zip(self.x, self.g_values)))
        self.b_sd = SpectralDistribution(data=dict(zip(self.x, self.b_values)))

    def get_wrgb_sd_array(self):
        return [self.w_sd, self.r_sd, self.g_sd, self.b_sd]

    def plot_rgb_distribution(self):
        fig, ax1 = pu.plot_1_graph(
            fontsize=20,
            figsize=(10, 8),
            graph_title="Spectral power distribution",
            graph_title_size=None,
            xlabel="Wavelength [nm]", ylabel="???",
            axis_label_size=None,
            legend_size=17,
            xlim=None,
            ylim=None,
            xtick=None,
            ytick=None,
            xtick_size=None, ytick_size=None,
            linewidth=5,
            return_figure=True)
        ax1.plot(self.x, self.r_values, color=pu.RED, label="R", alpha=0.6)
        ax1.plot(self.x, self.g_values, color=pu.GREEN, label="G", alpha=0.6)
        ax1.plot(self.x, self.b_values, color=pu.BLUE, label="B", alpha=0.6)
        ax1.plot(
            self.sd.wavelengths, self.sd.values, '--', color=(0.1, 0.1, 0.1),
            label="W", lw=2)
        plt.legend(loc='upper left')
        plt.show()
        plt.close(fig)


def create_display_spectrum_test():
    param_dict = dict(
        wavelengths=np.arange(360, 831),
        r_mean=625, r_dist=7.5, r_gain=50,
        g_mean=530, g_dist=7.5, g_gain=50,
        b_mean=460, b_dist=7.5, b_gain=50)
    display_spectral_distribution = DisplaySpectralDistribution(**param_dict)
    display_w_sd = display_spectral_distribution.get_wrgb_sd_array()[0]

    # display_sd.plot_rgb_distribution()
    cmfs = get_cie_2_1931_cmf()
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Spectral power distribution",
        graph_title_size=None,
        xlabel="Wavelength [nm]", ylabel="???",
        axis_label_size=None,
        legend_size=17,
        xlim=[340, 750],
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        return_figure=True)
    ax1.plot(
        display_w_sd.wavelengths, display_w_sd.values, '-',
        color=(0.1, 0.1, 0.1), label="Display (W=R+G+B)")
    ax1.plot(
        cmfs.wavelengths, cmfs.values[..., 0], '--', color=pu.RED,
        label="Color matching function(R)", lw=1.5)
    ax1.plot(
        cmfs.wavelengths, cmfs.values[..., 1], '--', color=pu.GREEN,
        label="Color matching function(G)", lw=1.5)
    ax1.plot(
        cmfs.wavelengths, cmfs.values[..., 2], '--', color=pu.BLUE,
        label="Color matching function(B)", lw=1.5)
    plt.legend(loc='upper right')
    plt.savefig(
        "./img/ds_sd_dist_7.5.png", bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close(fig)


def plot_display_gamut_test():
    r_mean = 625
    g_mean = 530
    b_mean = 460
    dist = 20
    w_param_dict = dict(
        wavelengths=np.arange(360, 831),
        r_mean=r_mean, r_dist=dist, r_gain=50,
        g_mean=g_mean, g_dist=dist, g_gain=50,
        b_mean=b_mean, b_dist=dist, b_gain=50)
    display_sd_obj = DisplaySpectralDistribution(**w_param_dict)
    display_sd_array = display_sd_obj.get_wrgb_sd_array()
    w_display_sd = display_sd_array[0]
    r_display_sd = display_sd_array[1]
    g_display_sd = display_sd_array[2]
    b_display_sd = display_sd_array[3]
    cmfs = get_cie_2_1931_cmf()
    w_xyY = calc_xyY_from_single_spectrum(
        src_sd=REFRECT_100P_SD, ref_sd=w_display_sd, cmfs=cmfs)
    r_xyY = calc_xyY_from_single_spectrum(
        src_sd=REFRECT_100P_SD, ref_sd=r_display_sd, cmfs=cmfs)
    g_xyY = calc_xyY_from_single_spectrum(
        src_sd=REFRECT_100P_SD, ref_sd=g_display_sd, cmfs=cmfs)
    b_xyY = calc_xyY_from_single_spectrum(
        src_sd=REFRECT_100P_SD, ref_sd=b_display_sd, cmfs=cmfs)

    white = w_xyY
    primaries = np.vstack((r_xyY, g_xyY, b_xyY, r_xyY))

    rate = 480 / 755.0 * 2
    xmin = 0.0
    xmax = 0.8
    ymin = 0.0
    ymax = 0.9

    # プロット用データ準備
    # ---------------------------------
    xy_image = get_chromaticity_image(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    cmf_xy = _get_cmfs_xy()

    bt709_gamut, _ = get_primaries(name=cs.BT709)
    bt2020_gamut, _ = get_primaries(name=cs.BT2020)
    dci_p3_gamut, _ = get_primaries(name=cs.P3_D65)
    xlim = (min(0, xmin), max(0.8, xmax))
    ylim = (min(0, ymin), max(0.9, ymax))

    ax1 = pu.plot_1_graph(fontsize=20 * rate,
                          figsize=((xmax - xmin) * 10 * rate,
                                   (ymax - ymin) * 10 * rate),
                          graph_title="CIE1931 Chromaticity Diagram",
                          graph_title_size=None,
                          xlabel=None, ylabel=None,
                          axis_label_size=None,
                          legend_size=18 * rate,
                          xlim=xlim, ylim=ylim,
                          xtick=[x * 0.1 + xmin for x in
                                 range(int((xlim[1] - xlim[0])/0.1) + 1)],
                          ytick=[x * 0.1 + ymin for x in
                                 range(int((ylim[1] - ylim[0])/0.1) + 1)],
                          xtick_size=17 * rate,
                          ytick_size=17 * rate,
                          linewidth=4 * rate,
                          minor_xtick_num=2,
                          minor_ytick_num=2)
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=3.5*rate, label=None)
    ax1.plot((cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
             '-k', lw=3.5*rate, label=None)
    ax1.plot(bt709_gamut[:, 0], bt709_gamut[:, 1],
             c=pu.RED, label="BT.709", lw=2*rate, alpha=0.8)
    ax1.plot(bt2020_gamut[:, 0], bt2020_gamut[:, 1],
             c=pu.YELLOW, label="BT.2020", lw=2*rate, alpha=0.8)
    ax1.plot(dci_p3_gamut[:, 0], dci_p3_gamut[:, 1],
             c=pu.BLUE, label="DCI-P3", lw=2*rate, alpha=0.8)
    ax1.plot(primaries[:, 0], primaries[:, 1],
             c='k', label="Display device", lw=2.75*rate)
    ax1.plot(
        D65_WHITE[0], D65_WHITE[1], 'x', c=pu.RED, label="D65", ms=15, mew=5)
    ax1.plot(
        white[0], white[1], 'x', c='k', label="White point", ms=15, mew=5)

    ax1.imshow(xy_image, extent=(xmin, xmax, ymin, ymax))
    plt.legend(loc='upper right')
    plt.savefig(f"./img/dist_{dist}.png", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # debug_func()
    # extrapolator_test()
    # create_display_spectrum_test()
    plot_display_gamut_test()
