# -*- coding: utf-8 -*-
"""
debug
==============

"""

# import standard libraries
import os
import pathlib

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


def img_file_read_float(filename):
    img_int = img_file_read(filename)
    img_float = img_int / 0xFFFF

    return img_float


def img_file_write(filename, img):
    """
    OpenCV の BGR 配列が怖いので並べ替えるwrapperを用意。
    """
    cv2.imwrite(filename, img[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 9])


def img_file_wirte_float_to_16bit(filename, img_float):
    img_int = np.uint16(np.round(np.clip(img_float, 0.0, 1.0) * 0xFFFF))
    img_file_write(filename, img_int)


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


def make_blog_result_image():
    img_list = [
        "./img/step_ramp_step_65.png", "./img/dark.png",
        "./img/middle.png", "./img/high.png", "./img/umi.png"]
    youtube_lut = read_LUT("./3DLUT/HDR10_to_BT709_YouTube_Rev03.cube")
    luminance_lut = read_LUT(
        "./3DLUT/LuminanceMap_for_ST2084_BT2020_D65_MapRange_100-4000nits_65x65x65.cube")
    bt2446_1000_lut = read_LUT(
        "./3DLUT/1000nits_v3__a_0.10_s_0.60_k1_0.69_k3_0.74_y_s_49.0_grid_65_gamma_2.4.cube")
    bt2446_4000_lut = read_LUT(
        "./3DLUT/4000nits_v3__a_0.10_s_0.60_k1_0.69_k3_0.74_y_s_41.0_grid_65_gamma_2.4.cube")
    dst_dir = "./blog_img"

    lut3d_list = [
        None, youtube_lut, luminance_lut,
        bt2446_1000_lut, bt2446_4000_lut]

    for src_path in img_list:
        path_info = pathlib.Path(src_path)
        base_name = path_info.stem
        ext = path_info.suffix

        # make names
        dst_name_original = os.path.join(dst_dir, base_name + ext)
        dst_name_youtube = os.path.join(dst_dir, base_name + "_youtube" + ext)
        dst_name_luminance_map = os.path.join(
            dst_dir, base_name + "_luminance" + ext)
        dst_name_bt2446_1000 = os.path.join(
            dst_dir, base_name + "_bt2446_1000" + ext)
        dst_name_bt2446_4000 = os.path.join(
            dst_dir, base_name + "_bt2446_4000" + ext)

        dst_name_list = [
            dst_name_original, dst_name_youtube, dst_name_luminance_map,
            dst_name_bt2446_1000, dst_name_bt2446_4000]

        # original
        src_img = img_file_read_float(src_path)

        # apply roop
        for lut, dst_name in zip(lut3d_list, dst_name_list):
            print(f"converting {dst_name}")
            if lut is not None:
                dst_img = lut.apply(src_img)
            else:
                dst_img = src_img.copy()

            img_file_wirte_float_to_16bit(dst_name, dst_img)


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

    # lut3d = read_LUT("./3DLUT/_HDR10_to_BT709_YouTube_Rev03.cube")
    # write_LUT(lut3d, "./3DLUT/HDR10_to_BT709_YouTube_Rev03.cube")
    make_blog_result_image()
