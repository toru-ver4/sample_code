# -*- coding: utf-8 -*-
"""
modify and save the xml file for icc profile.
============================================

"""

# import standard libraries
import os
import xml.etree.ElementTree as ET

# import third-party libraries
import numpy as np
from colour.models import ACES_CG_COLOURSPACE,\
    ACES_2065_1_COLOURSPACE, RGB_COLOURSPACES
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


# def make_images(gamma_float=3.0):
#     src_color_space = cs.ACES_AP0

#     img = tpg.img_read_as_float(
#         "./img/ColorChecker_All_ITU-R BT.709_D65_BT1886_Reverse.tiff")
#     img_linear = img ** 2.4
#     img_sRGB = tf.oetf(img_linear, tf.SRGB)
#     ap1_img_linear = RGB_to_RGB(
#         img_linear,
#         RGB_COLOURSPACES[cs.BT709], RGB_COLOURSPACES[src_color_space])
#     ap1_non_linear = ap1_img_linear ** (1/gamma_float)
#     tpg.img_wirte_float_as_16bit_int("./img/ap0_img.png", ap1_non_linear)
#     tpg.img_wirte_float_as_16bit_int("./img/sRGB.png", img_sRGB)


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


def get_parametric_curve_element(root):
    parametric_curve_element = None
    parent_tag = 'parametricCurveType'
    key_tag = 'ParametricCurve'
    for parent_element in root.iter(parent_tag):
        if parent_element.find(key_tag) is not None:
            parametric_curve_element = parent_element.find(key_tag)

    return parametric_curve_element


def get_rgbXYZ_element_list(root):
    rgbXYZ_element_list = [None, None, None]
    parent_tag = 'XYZType'
    key_tag = 'TagSignature'
    target_tag = 'XYZNumber'
    key_text_list = ['rXYZ', 'gXYZ', 'bXYZ']

    for idx, key_text in enumerate(key_text_list):
        for parent_element in root.iter(parent_tag):
            if parent_element.find(key_tag).text == key_text:
                rgbXYZ_element_list[idx] = parent_element.find(target_tag)

    return rgbXYZ_element_list


def get_chad_matrix_from_chad_mtx_element(chad_mtx_element):
    """
    Parameters
    ----------
    chad_mtx_element : xml.etree.ElementTree.Element
        An instance of the Element class. It indicate the chad tag.

    Returns
    -------
    ndarray
        chromatic adaptation matrix.

    Examples
    --------
    >>> tree = ET.parse("./aces_ap0_gm30.xml")
    >>> root = tree.getroot()
    >>> chad_mtx_element = get_chad_mtx_element(root)
    >>> get_chad_matrix_from_chad_mtx_element(chad_mtx_element)
    [[ 1.04788208  0.02958679 -0.00924683]
     [ 0.0229187   0.99047852  0.01507568]
     [-0.05021667 -0.01707458  0.75167847]]
    """
    text_data = chad_mtx_element.text
    text_mtx_list = text_data.strip().replace("\n", "").split()
    mtx = np.array([float(x) for x in text_mtx_list]).reshape((3, 3))

    return mtx


def set_chad_matrix_to_chad_mtx_element(mtx, chad_mtx_element):
    """
    Parameters
    ----------
    mtx : ndarray
        chromatic adaptation matrix
    chad_mtx_element : xml.etree.ElementTree.Element
        An instance of the Element class. It indicate the chad tag.

    Returns
    -------
    None

    Examples
    --------
    >>> tree = ET.parse("./aces_ap0_gm30.xml")
    >>> root = tree.getroot()
    >>> chad_mtx_element = get_chad_mtx_element(root)
    >>> print(chad_mtx_element.text)

                    1.04788208 0.02291870 -0.05021667
                    0.02958679 0.99047852 -0.01707458
                    -0.00924683 0.01507568 0.75167847

    >>> mtx = np.array([
    ...     [0.1, 0.2, 0.3],
    ...     [0.4, 0.5, 0.6],
    ...     [0.7, 0.8, 0.9]])
    >>> set_chad_matrix_to_chad_mtx_element(mtx, chad_mtx_element)
    >>> print(chad_mtx_element.text)

                    0.10000000 0.20000000 0.30000000
                    0.40000000 0.50000000 0.60000000
                    0.70000000 0.80000000 0.90000000

    """
    head_space = " " * 12
    foot_space = " " * 6
    buf = "\n"
    buf += f"{head_space}{mtx[0][0]:.8f} {mtx[0][1]:.8f} {mtx[0][2]:.8f}\n"
    buf += f"{head_space}{mtx[0][0]:.8f} {mtx[0][1]:.8f} {mtx[0][2]:.8f}\n"
    buf += f"{head_space}{mtx[0][0]:.8f} {mtx[0][1]:.8f} {mtx[0][2]:.8f}\n"
    buf += foot_space

    chad_mtx_element.text = buf


def get_parametric_curve_params_from_element(parameteric_curve_element):
    """
    Parameters
    ----------
    parameteric_curve_element : xml.etree.ElementTree.Element
        An instance of the Element class. It indicate the ParametricCurve tag.

    Returns
    -------
    function_type_str : str
        A string of function type.
    params : ndarray
        A one-dimensional parameter array.

    Examples
    --------
    >>> tree = ET.parse("./p3-2.xml")
    >>> root = tree.getroot()
    >>> parameteric_curve_element = get_parametric_curve_element(root)
    >>> function_type, params = get_parametric_curve_params_from_element(
    ...     parameteric_curve_element)
    >>> print(function_type)
    3
    >>> print(params)
    [ 2.39999390 0.94786072 0.05213928 0.07739258 0.04045105 ]
    """
    function_type_str = parameteric_curve_element.attrib['FunctionType']
    text_data = parameteric_curve_element.text
    param_str_list = text_data.strip().replace("\n", "").split()
    param_ndarray = np.array([float(x) for x in param_str_list])

    return function_type_str, param_ndarray


def set_parametric_curve_params_to_element(
        function_type_str, params, parameteric_curve_element):
    """
    Parameters
    ----------
    function_type_str : str
        A function type.
    params : ndarray
        An array of the function parameters.
    parameteric_curve_element : xml.etree.ElementTree.Element
        An instance of the Element class. It indicate the ParametricCurve tag.

    Returns
    -------
    None

    Examples
    --------
    >>> tree = ET.parse("./p3-2.xml")
    >>> root = tree.getroot()
    >>> parameteric_curve_element = get_parametric_curve_element(root)
    >>> function_type = "2"
    >>> params = np.array([2.4, 0.1, 0.2, 0.3])
    >>> set_parametric_curve_params_to_element(
    ...     function_type, params, parameteric_curve_element)
    >>> print(parameteric_curve_element.attrib)
    {'FunctionType': '2'}
    >>> print(parameteric_curve_element.text)

            2.39999390 0.10000000 0.20000000 0.30000000

    """
    parameteric_curve_element.attrib['FunctionType'] = function_type_str

    head_space = '\n' + " " * 8
    foot_space = " " * 6
    buf = head_space
    param_str_array = [f"{x:.8f}" for x in params]
    buf += " ".join(param_str_array) + '\n' + foot_space
    parameteric_curve_element.text = buf


def get_rgbXYZ_params_from_element(rgbXYZ_element_list):
    """
    Parameters
    ----------
    rgbXYZ_element_list : list
        list of xml.etree.ElementTree.Element.

    Returns
    -------
    ndarray
        A 3x3 matrix that convert from src color space to the PCS.

    Examples
    --------
    >>> tree = ET.parse("./p3-2.xml")
    >>> root = tree.getroot()
    >>> get_rgbXYZ_params_from_element(rgbXYZ_element_list)
    
    """


def xml_parse_test():
    tree = ET.parse("./p3-2.xml")
    root = tree.getroot()
    chad_mtx_element = get_chad_mtx_element(root)
    print(chad_mtx_element.text)
    get_chad_matrix_from_chad_mtx_element(chad_mtx_element)
    set_chad_matrix_to_chad_mtx_element(
        mtx=np.ones((3, 3)), chad_mtx_element=chad_mtx_element)
    print(chad_mtx_element.text)
    desc_element = get_desc_element(root)
    print(desc_element.text)
    cprt_element = get_cprt_element(root)
    print(cprt_element.text)
    profile_ver_element = get_icc_header_element(root, tag="ProfileVersion")
    print(profile_ver_element.text)
    color_space_element = get_icc_header_element(root, tag="DataColourSpace")
    print(color_space_element.text)
    parameteric_curve_element = get_parametric_curve_element(root)
    print(parameteric_curve_element.text)
    function_type, params = get_parametric_curve_params_from_element(
        parameteric_curve_element)
    print(function_type)
    print(params)

    function_type = "2"
    params = np.array([2.4, 0.1, 0.2, 0.3])
    set_parametric_curve_params_to_element(
        function_type, params, parameteric_curve_element)
    print(parameteric_curve_element.attrib)
    print(parameteric_curve_element.text)

    rgb_XYZ_element_list = get_rgbXYZ_element_list(root)
    for rgb_XYZ_element in rgb_XYZ_element_list:
        print(rgb_XYZ_element.attrib)

    tree.write("test_out.xml")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # main_func()
    xml_parse_test()
