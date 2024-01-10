# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import xy_to_XYZ
from colour import RGB_COLOURSPACES

# import my libraries
import test_pattern_generator2 as tpg
import color_space as cs
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def output_rgb_array_for_markdown_table(rgb_array, text_array):
    print_option_bak = np.get_printoptions()
    np.set_printoptions(precision=4)

    buf = "| title | R patch | G patch | B patch |\n"
    buf += "|:----:|:----:|:----:|:----:|\n"
    for rgb, text in zip(rgb_array, text_array):
        buf += f"| {text} | ({rgb[0][0]:.5f}, {rgb[0][1]:.5f}, {rgb[0][2]:.5f}) |"
        buf += f"({rgb[1][0]:.5f}, {rgb[1][1]:.5f}, {rgb[1][2]:.5f}) |"
        buf += f"({rgb[2][0]:.5f}, {rgb[2][1]:.5f}, {rgb[2][2]:.5f}) |\n"
    print(buf)
    np.set_printoptions(**print_option_bak)


def evaluate_quantization_error():
    xyY = tpg.generate_color_checker_xyY_value()
    xyY_rgb_patch = np.array([xyY[14], xyY[13], xyY[12]])
    large_xyz = xy_to_XYZ(xyY_rgb_patch)
    rgb_p3d65 = cs.large_xyz_to_rgb(large_xyz, cs.P3_D65)

    rgb_st2084_16bit = np.round(tf.oetf_from_luminance(rgb_p3d65 * 100, tf.ST2084) * 0x3FF)
    rgb_p3d65_after = tf.eotf_to_luminance(rgb_st2084_16bit / 0x3FF, tf.ST2084) / 100
    large_xyz_after = cs.rgb_to_large_xyz(rgb_p3d65_after, cs.P3_D65)
    print(large_xyz - large_xyz_after)


def create_expected_linear_value():
    img = tpg.img_read_as_float("./img/ARRI_LOG_C_ARRI_Wide_Gamut_4.png")
    # pos_list = [[1403, 402], [1308, 402], [1187, 402]]
    rgb_patch = np.array([img[402, 1403], img[402, 1308], img[402, 1187]])
    rgb_linear = tf.eotf_to_luminance(rgb_patch, tf.LOGC4) / 100
    large_xyz = cs.rgb_to_large_xyz(rgb_linear, cs.ALEXA_WIDE_GAMUT_4, cs.D65)
    rgb_p3d65 = cs.large_xyz_to_rgb(large_xyz, cs.P3_D65)
    rgb_bt2020 = cs.large_xyz_to_rgb(large_xyz, cs.BT2020)
    rgb_ap0 = cs.large_xyz_to_rgb(large_xyz, cs.ACES_AP0, xyz_white=cs.D65)
    rgb_ap1 = cs.large_xyz_to_rgb(large_xyz, cs.ACES_AP1, xyz_white=cs.D65)

    output_rgb_array_for_markdown_table(
        rgb_array=[rgb_p3d65, rgb_bt2020, rgb_ap1, rgb_ap0],
        text_array=["P3D65", "BT.2020", "ACES AP1", "ACES AP0"]
    )


def create_expected_sRGB_value():
    img = tpg.img_read_as_float("./img/ARRI_LOG_C_ARRI_Wide_Gamut_4.png")
    # pos_list = [[1403, 402], [1308, 402], [1187, 402]]
    rgb_patch = np.array([img[402, 1403], img[402, 1308], img[402, 1187]])
    rgb_linear = tf.eotf_to_luminance(rgb_patch, tf.LOGC4) / 100
    large_xyz = cs.rgb_to_large_xyz(rgb_linear, cs.ALEXA_WIDE_GAMUT_4, cs.D65)
    srgb_linear = cs.large_xyz_to_rgb(large_xyz, cs.sRGB)
    srgb_non_linear = tf.oetf(srgb_linear, tf.SRGB)

    output_rgb_array_for_markdown_table(
        rgb_array=[srgb_non_linear],
        text_array=["sRGB"]
    )


def conv_to_sRGB_value():
    img = tpg.img_read_as_float("./img/ARRI_LOG_C_ARRI_Wide_Gamut_4.png")
    # pos_list = [[1403, 402], [1308, 402], [1187, 402]]
    rgb_patch = img
    rgb_linear = tf.eotf_to_luminance(rgb_patch, tf.LOGC4) / 100
    large_xyz = cs.rgb_to_large_xyz(rgb_linear, cs.ALEXA_WIDE_GAMUT_4, cs.D65)
    srgb_linear = cs.large_xyz_to_rgb(large_xyz, cs.sRGB)
    srgb_non_linear = tf.oetf(srgb_linear, tf.SRGB)

    tpg.img_wirte_float_as_16bit_int(
        "./img/ARRI_LOG_C_ARRI_Wide_Gamut_4_to_sRGB.png", srgb_non_linear)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_expected_linear_value()
    # create_expected_sRGB_value()
    conv_to_sRGB_value()
    # evaluate_quantization_error()
