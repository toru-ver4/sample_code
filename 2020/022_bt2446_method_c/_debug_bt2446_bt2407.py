# -*- coding: utf-8 -*-
"""
debug
==============

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
import cv2
from colour import read_LUT, write_LUT, LUT3D
import matplotlib.pyplot as plt

# import my libraries
import transfer_functions as tf
import test_pattern_generator2 as tpg
import color_space as cs
import bt2446_method_c as bmc
import bt2047_gamut_mapping as bgm

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def img_file_read(filename):
    """
    OpenCV の BGR 配列が怖いので並べ替えるwrapperを用意。
    """
    img = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)

    if img is not None:
        return img[:, :, ::-1]
    else:
        return img


def img_file_write(filename, img):
    """
    OpenCV の BGR 配列が怖いので並べ替えるwrapperを用意。
    """
    cv2.imwrite(filename, img[:, :, ::-1])


def main_func():
    img_path = "./img/high.png"
    hdr_img_non_linear = bmc.read_img_and_to_float(img_path)
    hdr_img_linear = tf.eotf(hdr_img_non_linear, tf.ST2084)
    sdr_img_linear = bmc.bt2446_method_c_tonemapping(hdr_img_linear)

    tpg.preview_image(sdr_img_linear ** (1/2.4))

    sdr_709_liner = bgm.bt2407_gamut_mapping_for_rgb_linear(
        rgb_linear=sdr_img_linear,
        outer_color_space_name=cs.BT2020,
        inner_color_space_name=cs.BT709)
    tpg.preview_image(sdr_709_liner ** (1/2.4))


def get_youtube_tonemap_line():
    x_pq = np.linspace(0, 1, 1024)
    x_img = np.dstack([x_pq, x_pq, x_pq])
    lut3d = read_LUT("./luts/HDR10_to_BT709_YouTube_Rev03.cube")
    y = lut3d.apply(x_img)

    plt.plot(x_pq, y[..., 1].flatten())
    x = tf.eotf_to_luminance(x_pq, tf.ST2084)
    y = tf.eotf_to_luminance(y[..., 1].flatten(), tf.GAMMA24)
    out_data = np.dstack((x, y)).reshape((1024, 2))
    print(out_data)
    np.save("./youtube.npy", out_data)


def apply_bt2446_bt2407(
        src_color_space_name=cs.BT2020, tfc=tf.ST2084,
        alpha=0.15, sigma=0.5,
        hdr_ref_luminance=203, hdr_peak_luminance=1000,
        k1=0.8, k3=0.7, y_sdr_ip=60, bt2407_gamut_mapping=True):

    img = img_file_read("./_debug_img/SMPTE ST2084_ITU-R BT.2020_D65_1920x1080_rev04_type1.tiff")
    x_linear = tf.eotf(img / 0xFFFF, tf.ST2084)
    sdr_img_linear = bmc.bt2446_method_c_tonemapping(
         img=x_linear,
         src_color_space_name=src_color_space_name,
         tfc=tfc, alpha=alpha, sigma=sigma,
         hdr_ref_luminance=hdr_ref_luminance,
         hdr_peak_luminance=hdr_peak_luminance,
         k1=k1, k3=k3, y_sdr_ip=y_sdr_ip)
    if bt2407_gamut_mapping:
        sdr_img_linear = bgm.bt2407_gamut_mapping_for_rgb_linear(
            rgb_linear=sdr_img_linear,
            outer_color_space_name=cs.BT2020,
            inner_color_space_name=cs.BT709)
    sdr_img_nonlinear = sdr_img_linear ** (1/2.4)

    out_img = np.uint16(np.round(sdr_img_nonlinear * 0xFFFF))
    img_file_write("./_debug_img/sdr.tiff", out_img)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # mbl.make_bt2020_to_bt709_luts()
    # main_func()
    # get_youtube_tonemap_line()
    # apply_bt2446_bt2407(
    #     src_color_space_name=cs.BT2020, tfc=tf.ST2084,
    #     alpha=0.05, sigma=0.75,
    #     hdr_ref_luminance=203, hdr_peak_luminance=1000,
    #     k1=0.51, k3=0.75, y_sdr_ip=51.1, bt2407_gamut_mapping=True)

    lut3d = read_LUT("./3DLUT/_HDR10_to_BT709_YouTube_Rev03.cube")
    write_LUT(lut3d, "./3DLUT/HDR10_to_BT709_YouTube_Rev03.cube")