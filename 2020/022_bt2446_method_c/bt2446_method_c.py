# -*- coding: utf-8 -*-
"""
Title
==============

Description.

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
import cv2
from colour import RGB_to_XYZ, XYZ_to_RGB, RGB_COLOURSPACES,\
    XYZ_to_Lab, Lab_to_LCHab, LCHab_to_Lab, Lab_to_XYZ, XYZ_to_xyY, xyY_to_XYZ
from colour.models import BT2020_COLOURSPACE
import matplotlib.pyplot as plt

# import my libraries
import transfer_functions as tf
import test_pattern_generator2 as tpg
import color_space as cs
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def imread_16bit_to_float(file_name):
    img = cv2.imread(
        file_name, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    return img[..., ::-1] / 0xFFFF


def read_img_and_to_float(path):
    img = cv2.imread(
        path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[..., ::-1]
    max_value = np.iinfo(img.dtype).max
    img = img / max_value

    return img


def apply_cross_talk_matrix(img, alpha=0.2):
    mtx = np.array([
        [1 - 2 * alpha, alpha, alpha],
        [alpha, 1 - 2 * alpha, alpha],
        [alpha, alpha, 1 - 2 * alpha]
    ])
    out_img = np.zeros_like(img)
    out_img[..., 0] = img[..., 0] * mtx[0][0] + img[..., 1] * mtx[0][1] + img[..., 2] * mtx[0][2]
    out_img[..., 1] = img[..., 1] * mtx[1][0] + img[..., 1] * mtx[1][1] + img[..., 2] * mtx[1][2]
    out_img[..., 2] = img[..., 2] * mtx[2][0] + img[..., 1] * mtx[2][1] + img[..., 2] * mtx[2][2]

    return out_img


def apply_inverse_cross_talk_matrix(img, alpha=0.2):
    mtx = np.array([
        [1-alpha, -alpha, -alpha],
        [-alpha, 1-alpha, -alpha],
        [-alpha, -alpha, 1-alpha]
    ]) / (1 - 3 * alpha)
    out_img = np.zeros_like(img)
    out_img[..., 0] = img[..., 0] * mtx[0][0] + img[..., 1] * mtx[0][1] + img[..., 2] * mtx[0][2]
    out_img[..., 1] = img[..., 1] * mtx[1][0] + img[..., 1] * mtx[1][1] + img[..., 2] * mtx[1][2]
    out_img[..., 2] = img[..., 2] * mtx[2][0] + img[..., 1] * mtx[2][1] + img[..., 2] * mtx[2][2]

    return out_img


def rgb_to_xyz_in_hdr_space(rgb, color_space_name):
    xyz = RGB_to_XYZ(
        rgb, cs.D65, cs.D65,
        RGB_COLOURSPACES[color_space_name].RGB_to_XYZ_matrix)

    return xyz


def apply_chroma_correction(
        xyz, sigma, src_color_space_name, tfc,
        hdr_ref_luminance=203, hdr_peak_luminance=1000):
    lmax, lref = calc_lmax_lref(
        color_space_name=src_color_space_name, tfc=tfc,
        hdr_ref_luminance=hdr_ref_luminance,
        hdr_peak_luminance=hdr_peak_luminance)
    lch = Lab_to_LCHab(XYZ_to_Lab(xyz))
    ll = lch[..., 0]
    cc = lch[..., 1]
    hh = lch[..., 2]

    cor_idx = (ll > lref)
    cc[cor_idx] = cc[cor_idx]\
        * (1 - sigma * ((ll[cor_idx] - lref)/(lmax - lref)))
    cc[cc < 0] = 0

    lch_cor = np.dstack((ll, cc, hh)).reshape(lch.shape)
    xyz_cor = Lab_to_XYZ(LCHab_to_Lab(lch_cor))

    return xyz_cor


def mono2color(value):
    return np.array([value for x in range(3)])


def calc_lmax_lref(
        color_space_name=cs.BT2020, tfc=tf.ST2084,
        hdr_ref_luminance=203, hdr_peak_luminance=1000):
    luminance_rgb = np.array(
        [mono2color(hdr_ref_luminance), mono2color(hdr_peak_luminance)])
    luminance_rgb_normalized = luminance_rgb / tf.PEAK_LUMINANCE[tfc]
    xyz = RGB_to_XYZ(
        luminance_rgb_normalized, cs.D65, cs.D65,
        RGB_COLOURSPACES[color_space_name].RGB_to_XYZ_matrix)
    lab = XYZ_to_Lab(xyz)
    lref = lab[0, 0]
    lmax = lab[1, 0]

    return lmax, lref


def calc_tonemapping_parameters(
        k1=0.8, k3=0.7, y_sdr_ip=60, y_hdr_ref=203):
    """
    calculate tonemapping parameters

    Parameters
    ----------
    k1 : float
        k1. the range is from 0.0 to 1.0?
    k3 : float
        k3. the range is from 0.0 to 1.0?
    y_sdr_ip : float
        luminance of the output SDR image at the inflection point.
    y_hdr_ref : float
        luminance of the input HDR image at the reference white.

    Returns
    -------
    y_hdr_ip : float
        luminance of the input HDR image at the inflection point.
    y_sdr_wp : float
        luminance of the output SDR image at the white point.
    k2 : float
        k2.
    k4 : float
        k4.
    """
    y_hdr_ip = y_sdr_ip / k1
    k2 = k1 * (1 - k3) * y_hdr_ip
    k4 = k1 * y_hdr_ip - k2 * np.log(1 - k3)
    y_sdr_wp = k2 * np.log(y_hdr_ref / y_hdr_ip - k3) + k4

    return y_hdr_ip, y_sdr_wp, k2, k4


def bt2446_method_c_tonemapping_core(
        x, k1=0.8, k3=0.7, y_sdr_ip=60, y_hdr_ref=203):
    """
    calculate tonemapping parameters

    Parameters
    ----------
    x : array_like
        input hdr linear data. the unit must be luminance [nits].
    k1 : float
        k1. the range is from 0.0 to 1.0?
    k3 : float
        k3. the range is from 0.0 to 1.0?
    y_sdr_ip : float
        luminance of the output SDR image at the inflection point.
    y_hdr_ref : float
        luminance of the input HDR image at the reference white.

    Returns
    -------
    y : array_like
        sdr linear data. the unit is luminance [nits].
    """
    y_hdr_ip, y_sdr_wp, k2, k4 = calc_tonemapping_parameters(
        k1=k1, k3=k3, y_sdr_ip=y_sdr_ip, y_hdr_ref=y_hdr_ref)
    y_hdr_ip = y_sdr_ip / k1
    y = np.where(
        x < y_hdr_ip,
        x * k1,
        k2 * np.log(x / y_hdr_ip - k3) + k4)

    return y


def draw_ip_wp_annotation(
        x_min, y_min, ax1, k1, y_hdr_ip, y_hdr_ref, y_sdr_wp, fontsize=20):
    # annotation
    arrowprops = dict(
        facecolor='#333333', shrink=0.0, headwidth=8, headlength=10,
        width=1, alpha=0.5)
    ax1.annotate(
        "Y_HDR,ip", xy=(y_hdr_ip, y_min), xytext=(500, y_min * 5.1),
        xycoords='data', textcoords='data', ha='left', va='bottom',
        arrowprops=arrowprops, fontsize=fontsize)
    ax1.annotate(
        "Y_HDR,Ref", xy=(y_hdr_ref, y_min), xytext=(500, y_min * 2),
        xycoords='data', textcoords='data', ha='left', va='bottom',
        arrowprops=arrowprops, fontsize=fontsize)
    ax1.annotate(
        "K1 * Y_HDR,ip", xy=(x_min, k1 * y_hdr_ip), xytext=(x_min * 4, 110),
        xycoords='data', textcoords='data', ha='left', va='bottom',
        arrowprops=arrowprops, fontsize=fontsize)
    ax1.annotate(
        "Y_SDR,wp", xy=(x_min, y_sdr_wp), xytext=(x_min * 4, 170),
        xycoords='data', textcoords='data', ha='left', va='bottom',
        arrowprops=arrowprops, fontsize=fontsize)


def tone_map_plot_test(k1=0.8, k3=0.7, y_sdr_ip=60, y_hdr_ref=203):
    x_min = 1
    y_min = 1
    y_hdr_ip, y_sdr_wp, k2, k4 = calc_tonemapping_parameters(
        k1=k1, k3=k3, y_sdr_ip=y_sdr_ip, y_hdr_ref=y_hdr_ref)
    x = np.linspace(0, 10000, 1024)
    y = bt2446_method_c_tonemapping_core(
        x, k1=k1, k3=k3, y_sdr_ip=y_sdr_ip, y_hdr_ref=y_hdr_ref)
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="BT.2446 Method C",
        xlabel="HDR Luminance [cd/m2]",
        ylabel="SDR Luminance [cd/m2]",
        xlim=(x_min, 10000),
        ylim=(y_min, 250),
        linewidth=3,
        return_figure=True)
    pu.log_scale_settings(ax1)
    ax1.plot(x, y)
    # plt.show()

    # annotation
    draw_ip_wp_annotation(
        x_min, y_min, ax1, k1, y_hdr_ip, y_hdr_ref, y_sdr_wp)

    # auxiliary line
    ax1.plot(
        [y_hdr_ip, y_hdr_ip, x_min], [y_min, k1 * y_hdr_ip, k1 * y_hdr_ip],
        'k--', lw=2, c='#555555')
    ax1.plot(
        [y_hdr_ref, y_hdr_ref, x_min], [y_min, y_sdr_wp, y_sdr_wp],
        'k--', lw=2, c='#555555')

    # fname = f"./figures/k1_{k1:.2f}_k3_{k3:.2f}_y_sdr_ip_{y_sdr_ip:.1f}.png"
    plt.show()
    # plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)


def experimental_func(
        src_color_space_name=cs.BT2020, tfc=tf.ST2084,
        alpha=0.15, sigma=0.5, hdr_ref_luminance=203, hdr_peak_luminance=1000,
        k1=0.8, k3=0.7, y_sdr_ip=60,
        test_img_path="./img/step_ramp_step_65.png"):
    img = read_img_and_to_float(test_img_path)
    img_linear = tf.eotf(img, tfc)
    img_desturated = apply_cross_talk_matrix(img=img_linear, alpha=alpha)
    # tpg.preview_image(img_desturated ** (1/2.4))
    xyz_hdr = rgb_to_xyz_in_hdr_space(
        rgb=img_desturated, color_space_name=src_color_space_name)
    xyz_hdr_cor = apply_chroma_correction(
        xyz=xyz_hdr, sigma=sigma, src_color_space_name=src_color_space_name,
        tfc=tfc, hdr_ref_luminance=hdr_ref_luminance,
        hdr_peak_luminance=hdr_peak_luminance)
    xyY_hdr_cor = XYZ_to_xyY(xyz_hdr_cor)

    y_hdr = xyY_hdr_cor[..., 2] * tf.PEAK_LUMINANCE[tfc]
    y_sdr = bt2446_method_c_tonemapping_core(
        y_hdr, k1=k1, k3=k3, y_sdr_ip=y_sdr_ip, y_hdr_ref=hdr_ref_luminance)
    y_sdr = np.clip(y_sdr, 0.0, tf.PEAK_LUMINANCE[tf.GAMMA24])\
        / tf.PEAK_LUMINANCE[tf.GAMMA24]

    xyY_sdr_cor = xyY_hdr_cor.copy()
    xyY_sdr_cor[..., 2] = y_sdr
    xyz_sdr_cor = xyY_to_XYZ(xyY_sdr_cor)

    rgb_sdr_linear = XYZ_to_RGB(
        xyz_sdr_cor, cs.D65, cs.D65,
        RGB_COLOURSPACES[src_color_space_name].XYZ_to_RGB_matrix)
    rgb_sdr_linear = apply_inverse_cross_talk_matrix(
        img=rgb_sdr_linear, alpha=alpha)
    print(rgb_sdr_linear.shape)
    rgb_sdr_linear = np.clip(rgb_sdr_linear, 0.0, 1.0)

    # tone_map_plot_test(
    #     k1=k1, k3=k3, y_sdr_ip=y_sdr_ip, y_hdr_ref=hdr_ref_luminance)

    tpg.preview_image(rgb_sdr_linear ** (1/2.4))


def bt2446_method_c_tonemapping(
        img, src_color_space_name=cs.BT2020, tfc=tf.ST2084,
        alpha=0.15, sigma=0.5, hdr_ref_luminance=203, hdr_peak_luminance=1000,
        k1=0.8, k3=0.7, y_sdr_ip=60):
    """
    Apply tonemapping function described in BT.2446 Method C

    Parameters
    ----------
    img : array_like
        hdr linear image data.
        img is should be linear. the unit of the img should be nits.
    src_color_space_name : str
        the name of the color space of the src image.
    tfc : str
        the name of the transfer characteristics of the src image.
        it is used for calculation of the eotf-peak-luminance.
    alpha : float
        parameter of chrosstalk matrix used to reduce the chroma
        before applying the tonecurve.
    sigma : float
        parameter of chroma reduction.
    hdr_ref_luminance : int
        luminance of the HDR reference white.
    hdr_peak_luminance : int
        peak luminance of the src image. This parameter is used for
        chroma correction
    k1 : float
        parameter for tonecurve
    k3 : float
        parameter for tonecurve
    y_sdr_ip : float
        parameter for tonecurve

    Returns
    -------
    array_like
        sdr linear data. data range is 0.0 -- 1.0.
    """
    img_desturated = apply_cross_talk_matrix(img=img, alpha=alpha)
    xyz_hdr = rgb_to_xyz_in_hdr_space(
        rgb=img_desturated, color_space_name=src_color_space_name)
    xyz_hdr_cor = apply_chroma_correction(
        xyz=xyz_hdr, sigma=sigma, src_color_space_name=src_color_space_name,
        tfc=tfc, hdr_ref_luminance=hdr_ref_luminance,
        hdr_peak_luminance=hdr_peak_luminance)
    xyY_hdr_cor = XYZ_to_xyY(xyz_hdr_cor)

    y_hdr = xyY_hdr_cor[..., 2] * tf.PEAK_LUMINANCE[tfc]
    y_sdr = bt2446_method_c_tonemapping_core(
        y_hdr, k1=k1, k3=k3, y_sdr_ip=y_sdr_ip, y_hdr_ref=hdr_ref_luminance)
    y_sdr = np.clip(y_sdr, 0.0, tf.PEAK_LUMINANCE[tf.GAMMA24])\
        / tf.PEAK_LUMINANCE[tf.GAMMA24]

    xyY_sdr_cor = xyY_hdr_cor.copy()
    xyY_sdr_cor[..., 2] = y_sdr
    xyz_sdr_cor = xyY_to_XYZ(xyY_sdr_cor)

    rgb_sdr_linear = XYZ_to_RGB(
        xyz_sdr_cor, cs.D65, cs.D65,
        RGB_COLOURSPACES[src_color_space_name].XYZ_to_RGB_matrix)
    rgb_sdr_linear = apply_inverse_cross_talk_matrix(
        img=rgb_sdr_linear, alpha=alpha)
    rgb_sdr_linear = np.clip(rgb_sdr_linear, 0.0, 1.0)

    return rgb_sdr_linear


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    experimental_func(
        src_color_space_name=cs.BT2020, tfc=tf.ST2084,
        alpha=0.15, sigma=0.5, hdr_ref_luminance=203, hdr_peak_luminance=1000,
        k1=0.89, k3=0.72, y_sdr_ip=45)
