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
    XYZ_to_Lab, Lab_to_LCHab, LCHab_to_Lab, Lab_to_XYZ
from colour.models import BT2020_COLOURSPACE

# import my libraries
import transfer_functions as tf
import test_pattern_generator2 as tpg
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def imread_16bit_to_float(file_name):
    img = cv2.imread(
        file_name, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    return img[..., ::-1] / 0xFFFF


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
    cc[cor_idx] = cc[cor_idx] * (1 - sigma * ((ll[cor_idx] - lref)/(lmax - lref)))
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


def experimental_func(
        src_color_space_name=cs.BT2020, alpha=0.15, sigma=0.5,
        tfc=tf.ST2084, hdr_ref_luminance=203, hdr_peak_luminance=1000):
    img = imread_16bit_to_float("./img/step_ramp.png")
    img_linear = tf.eotf(img, tf.GAMMA24)
    img_desturated = apply_cross_talk_matrix(img=img_linear, alpha=alpha)
    # tpg.preview_image(img_desturated ** (1/2.4))
    xyz_hdr = rgb_to_xyz_in_hdr_space(
        rgb=img_desturated, color_space_name=src_color_space_name)
    xyz_hdr_cor = apply_chroma_correction(
        xyz=xyz_hdr, sigma=sigma, src_color_space_name=src_color_space_name,
        tfc=tfc, hdr_ref_luminance=hdr_ref_luminance,
        hdr_peak_luminance=hdr_peak_luminance)

    rgb_sdr = XYZ_to_RGB(
        xyz_hdr_cor, cs.D65, cs.D65,
        RGB_COLOURSPACES[src_color_space_name].XYZ_to_RGB_matrix)

    tpg.preview_image(rgb_sdr ** (1/2.4))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    experimental_func(
        src_color_space_name=cs.BT2020, alpha=0.2, sigma=0.5,
        tfc=tf.ST2084, hdr_ref_luminance=203, hdr_peak_luminance=1000)
