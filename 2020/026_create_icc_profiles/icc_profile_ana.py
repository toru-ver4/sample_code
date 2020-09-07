# -*- coding: utf-8 -*-
"""

============================================

"""

# import standard libraries
import os
import xml.etree.ElementTree as ET

# import third-party libraries
import numpy as np
from colour.models import ACES_CG_COLOURSPACE, ACES_2065_1_COLOURSPACE, RGB_COLOURSPACES
from colour import RGB_to_RGB

# import my libraries
import color_space as cs
import transfer_functions as tf
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

const_dci_p3_xy = [[0.680, 0.320],
                   [0.265, 0.690],
                   [0.150, 0.060]]

const_aces_ap0_xy = [[0.73470, 0.26530],
                     [0.00000, 1.00000],
                     [0.00010, -0.07700]]

const_d50_large_xyz = [96.422, 100.000, 82.521]
const_d65_large_xyz = [95.047, 100.000, 108.883]


def make_images(gamma_float=3.0):
    src_color_space = cs.ACES_AP0

    img = tpg.img_read_as_float(
        "./img/ColorChecker_All_ITU-R BT.709_D65_BT1886_Reverse.tiff")
    img_linear = img ** 2.4
    img_sRGB = tf.oetf(img_linear, tf.SRGB)
    ap1_img_linear = RGB_to_RGB(
        img_linear,
        RGB_COLOURSPACES[cs.BT709], RGB_COLOURSPACES[src_color_space])
    ap1_non_linear = ap1_img_linear ** (1/gamma_float)
    tpg.img_wirte_float_as_16bit_int("./img/ap0_img.png", ap1_non_linear)
    tpg.img_wirte_float_as_16bit_int("./img/sRGB.png", img_sRGB)


def main_func():
    # rgb_to_xyz = RGB_COLOURSPACES[src_color_space].RGB_to_XYZ_matrix
    rgb_to_xyz = cs.calc_rgb_to_xyz_matrix(
        const_aces_ap0_xy, const_d65_large_xyz)
    chromatic_adaptation = np.array(
        [[1.04788208, 0.02291870, -0.05021667],
         [0.02958679, 0.99047852, -0.01707458],
         [-0.00924683, 0.01507568, 0.75167847]])

    print(chromatic_adaptation.dot(rgb_to_xyz))

    gamma_float = 3.0
    gamma = int(gamma_float * 0x8000 + 0.5)
    print(f"gamma = {gamma / 0x8000:.16f}")

    make_images(gamma_float=gamma_float)


def get_icc_tag_element(
        root, parent_tag='s15Fixed16ArrayType',
        key_tag="TagSignature", key_text='chad',
        target_tag='Array'):
    icc_tag_element = None
    for parent_element in root.iter(parent_tag):
        if parent_element.find(key_tag).text == key_text:
            icc_tag_element = parent_element.find(target_tag)

    return icc_tag_element


def get_icc_header_element(root, tag="ProfileVersion"):
    header_element = None
    for element in root.iter(tag):
        header_element = element
        break

    return header_element


def get_chad_mtx_element(root):
    return get_icc_tag_element(
        root, parent_tag="s15Fixed16ArrayType",
        key_tag="TagSignature", key_text="chad", target_tag="Array")


def get_desc_element(root):
    return get_icc_tag_element(
        root, parent_tag="multiLocalizedUnicodeType",
        key_tag="TagSignature", key_text="desc", target_tag="LocalizedText")


def get_cprt_element(root):
    return get_icc_tag_element(
        root, parent_tag="multiLocalizedUnicodeType",
        key_tag="TagSignature", key_text="cprt", target_tag="LocalizedText")


def xml_parse_test():
    tree = ET.parse("./aces_ap0_gm30.xml")
    root = tree.getroot()
    chad_mtx_element = get_chad_mtx_element(root)
    print(chad_mtx_element.text)
    desc_element = get_desc_element(root)
    print(desc_element.text)
    cprt_element = get_cprt_element(root)
    print(cprt_element.text)
    profile_ver_element = get_icc_header_element(root, tag="ProfileVersion")
    print(profile_ver_element.text)
    color_space_element = get_icc_header_element(root, tag="DataColourSpace")
    print(color_space_element.text)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # main_func()
    xml_parse_test()
