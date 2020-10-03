# -*- coding: utf-8 -*-
"""
modify and save the xml file for icc profile.
============================================

"""

# import standard libraries
import os
import xml.etree.ElementTree as ET
from datetime import datetime

# import third-party libraries
import numpy as np

# import my libraries
import icc_profile_calc_param as ipcp
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


RENDERING_INTENT_ABSOLUTE = "Absolute Colorimetric"
RENDERING_INTENT_RELATIVE = "Relative Colorimetric"
RENDERING_INTENT_PERCEPTUAL = "Perceptual"
RENDERING_INTENT_SATURATION = "Saturation"


def get_value_from_specific_header_tag(root, header_name):
    """
    Example
    -------
    >>> tree = ET.parse("./aces_ap0_gm30.xml")
    >>> root = tree.getroot()
    >>> get_value_from_specific_header_tag(root, "RenderingIntent")
    "Perceptual"
    """
    header_value = None
    for element in root.iter(header_name):
        header_value = element.text

    return header_value


def set_value_to_specific_header_tag(root, header_name, value):
    """
    Example
    -------
    >>> tree = ET.parse("./aces_ap0_gm30.xml")
    >>> root = tree.getroot()
    >>> set_value_to_specific_header_tag(
    ...     root, "RenderingIntent", "Absolute Colorimetric")
    >>> get_value_from_specific_header_tag(root, "RenderingIntent")
    "Absolute Colorimetric"
    """
    for element in root.iter(header_name):
        element.text = value


def create_current_date_str():
    """
    create "YYYY-MM-DDTHH:MM:SS" format str.
    """
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


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


def get_wtpt_element(root):
    return get_icc_tag_element(
        root, parent_tag="XYZType",
        key_tag="TagSignature", key_text="wtpt", target_tag="XYZNumber")


def get_lumi_element(root):
    return get_icc_tag_element(
        root, parent_tag="XYZType",
        key_tag="TagSignature", key_text="lumi", target_tag="XYZNumber")


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
    buf += f"{head_space}{mtx[1][0]:.8f} {mtx[1][1]:.8f} {mtx[1][2]:.8f}\n"
    buf += f"{head_space}{mtx[2][0]:.8f} {mtx[2][1]:.8f} {mtx[2][2]:.8f}\n"
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
        '0' : simple power function
        '3' : sRGB like function
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
    >>> rgb_XYZ_element_list = get_rgbXYZ_element_list(root)
    >>> get_rgbXYZ_params_from_element(rgbXYZ_element_list)
    [[ 0.51512146  0.29197693  0.15710449]
     [ 0.24119568  0.69224548  0.0665741 ]
     [-0.00105286  0.04188538  0.78407288]]
    """
    src2pcs_mtx = np.zeros((3, 3))

    for idx, rgb_XYZ_element in enumerate(rgbXYZ_element_list):
        src2pcs_mtx[0][idx] = float(rgb_XYZ_element.attrib['X'])
        src2pcs_mtx[1][idx] = float(rgb_XYZ_element.attrib['Y'])
        src2pcs_mtx[2][idx] = float(rgb_XYZ_element.attrib['Z'])

    return np.array(src2pcs_mtx)


def set_rgbXYZ_params_to_element(src2pcs_mtx, rgb_XYZ_element_list):
    """
    Parameters
    ----------
    src2pcs_mtx : ndarray
        A 3x3 matrix that convert from src color space to the PCS.
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
    >>> rgb_XYZ_element_list = get_rgbXYZ_element_list(root)
    >>> src2pcs_mtx = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    >>> set_rgbXYZ_params_to_element(src2pcs_mtx, rgb_XYZ_element_list)
    >>> for rgb_XYZ_element in rgb_XYZ_element_list:
    ...     print(rgb_XYZ_element.attrib)
    {'X': '0.00000000', 'Y': '3.00000000', 'Z': '6.00000000'}
    {'X': '1.00000000', 'Y': '4.00000000', 'Z': '7.00000000'}
    {'X': '2.00000000', 'Y': '5.00000000', 'Z': '8.00000000'}
    """
    for idx, rgb_XYZ_element in enumerate(rgb_XYZ_element_list):
        rgb_XYZ_element.attrib['X'] = f"{src2pcs_mtx[0][idx]:.08f}"
        rgb_XYZ_element.attrib['Y'] = f"{src2pcs_mtx[1][idx]:.08f}"
        rgb_XYZ_element.attrib['Z'] = f"{src2pcs_mtx[2][idx]:.08f}"


def get_lumi_params_from_element(lumi_element):
    """
    Parameters
    ----------
    lumi_element : xml.etree.ElementTree.Element
        elemnent of the lumi tag.

    Returns
    -------
    float
        luminance of the reference white.

    Examples
    --------
    >>> tree = ET.parse("./p3-2.xml")
    >>> root = tree.getroot()
    >>> lumi_element = get_lumi_element(root)
    >>> get_lumi_params_from_element(lumi_element)
    100.0
    """
    return float(lumi_element.attrib['Y'])


def set_lumi_params_to_element(luminance, lumi_element):
    """
    Parameters
    ----------
    luminance : float
        luminance of the reference white.
    lumi_element : xml.etree.ElementTree.Element
        elemnent of the lumi tag.

    Returns
    -------
    None

    Examples
    --------
    >>> tree = ET.parse("./p3-2.xml")
    >>> root = tree.getroot()
    >>> lumi_element = get_lumi_element(root)
    >>> set_lumi_params_to_element(333.3, lumi_element)
    >>> get_lumi_params_from_element(lumi_element)
    333.3
    """
    lumi_element.attrib['Y'] = f"{luminance:.8f}"


def get_wtpt_params_from_element(wtpt_element):
    """
    Parameters
    ----------
    wtpt_element : xml.etree.ElementTree.Element
        elemnent of the wtpt tag.

    Returns
    -------
    ndarray
        white point in large XYZ format.

    Examples
    --------
    >>> tree = ET.parse("./p3-2.xml")
    >>> root = tree.getroot()
    >>> wtpt_element = get_wtpt_element(root)
    >>> get_wtpt_params_from_element(wtpt_element)
    [ 0.96420288  1.          0.8249054 ]
    """
    large_xyz = np.array(
        [float(wtpt_element.attrib['X']),
         float(wtpt_element.attrib['Y']),
         float(wtpt_element.attrib['Z'])])
    return large_xyz


def set_wtpt_params_to_element(wtpt, wtpt_element):
    """
    Parameters
    ----------
    wtpt : ndarray
        white point in large XYZ format.
    wtpt_element : xml.etree.ElementTree.Element
        elemnent of the wtpt tag.

    Returns
    -------
    None

    Examples
    --------
    >>> tree = ET.parse("./p3-2.xml")
    >>> root = tree.getroot()
    >>> wtpt_element = get_wtpt_element(root)
    >>> wtpt = np.array([0.9, 1.0, 0.8])
    >>> set_wtpt_params_to_element(wtpt, wtpt_element)
    >>> get_wtpt_params_from_element(wtpt_element)
    [ 0.90000000  1.0000000  0.80000000 ]
    """
    wtpt_element.attrib['X'] = f"{wtpt[0]:.8f}"
    wtpt_element.attrib['Y'] = f"{wtpt[1]:.8f}"
    wtpt_element.attrib['Z'] = f"{wtpt[2]:.8f}"


def xml_parse_test():
    tree = ET.parse("./icc_profile_sample/p3-2.xml")
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
    src2pcs_mtx = get_rgbXYZ_params_from_element(rgb_XYZ_element_list)
    print(src2pcs_mtx)
    src2pcs_mtx = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    set_rgbXYZ_params_to_element(src2pcs_mtx, rgb_XYZ_element_list)
    for rgb_XYZ_element in rgb_XYZ_element_list:
        print(rgb_XYZ_element.attrib)

    ri = get_value_from_specific_header_tag(root, "RenderingIntent")
    print(ri)
    set_value_to_specific_header_tag(
        root, "RenderingIntent", RENDERING_INTENT_PERCEPTUAL)
    current_time_str = create_current_date_str()
    print(current_time_str)

    wtpt_element = get_wtpt_element(root)
    print(wtpt_element.attrib)

    lumi_element = get_lumi_element(root)
    print(get_lumi_params_from_element(lumi_element))
    set_lumi_params_to_element(1000.0, lumi_element)

    wtpt_element = get_wtpt_element(root)
    print(wtpt_element.attrib)
    print(get_wtpt_params_from_element(wtpt_element))
    wtpt = np.array([0.123, 0.46, 0.88])
    set_wtpt_params_to_element(wtpt, wtpt_element)

    tree.write("test_out.xml")


def create_profle_header(root):
    """
    create the profile header.
    """
    set_value_to_specific_header_tag(
        root, "PreferredCMMType", "ADBE")
    set_value_to_specific_header_tag(
        root, "ProfileVersion", "4.30")
    set_value_to_specific_header_tag(
        root, "CreationDateTime", create_current_date_str())
    set_value_to_specific_header_tag(
        root, "PrimaryPlatform", "MSFT")
    set_value_to_specific_header_tag(
        root, "DeviceManufacturer", "")
    set_value_to_specific_header_tag(
        root, "DeviceModel", "")
    set_value_to_specific_header_tag(
        root, "ProfileCreator", "")
    set_value_to_specific_header_tag(
        root, "ProfileID", "")


def create_simple_power_gamma_profile(
        gamma=2.4, src_white=np.array([0.3127, 0.3290]),
        src_primaries=np.array([[0.680, 0.320], [0.265, 0.690], [0.15, 0.06]]),
        desc_str="Gamam=2.4_DCI-P3_D65",
        cprt_str="Copyright 2020 HOGEHOGE Corp.",
        output_name="Gamam=2.4_DCI-P3_D65.xml"):
    """
    create simple profile.
    gamma function must be "y = x ** gamma" format.
    """
    tree = ET.parse("./icc_profile_sample/base_profile_v4.xml")
    root = tree.getroot()

    # Profile header
    create_profle_header(root)

    # Tagged element data
    desc_element = get_desc_element(root)
    desc_element.text = desc_str

    cprt_element = get_cprt_element(root)
    cprt_element.text = cprt_str

    chad_mtx = ipcp.calc_chromatic_adaptation_matrix(
        src_white=src_white, dst_white=ipcp.PCS_D50)
    chad_mtx_element = get_chad_mtx_element(root)
    set_chad_matrix_to_chad_mtx_element(
        mtx=chad_mtx, chad_mtx_element=chad_mtx_element)

    lumi_element = get_lumi_element(root)
    set_lumi_params_to_element(
        luminance=100.0, lumi_element=lumi_element)

    wtpt_element = get_wtpt_element(root)
    set_wtpt_params_to_element(
        wtpt=ipcp.PCS_D50_XYZ, wtpt_element=wtpt_element)

    rgbXYZ_element_list = get_rgbXYZ_element_list(root)
    src2pcs_mtx = ipcp.calc_rgb_to_xyz_mtx_included_chad_mtx(
        rgb_primaries=src_primaries,
        src_white=src_white, dst_white=ipcp.PCS_D50)
    set_rgbXYZ_params_to_element(
        src2pcs_mtx=src2pcs_mtx, rgb_XYZ_element_list=rgbXYZ_element_list)

    parametric_curve_element = get_parametric_curve_element(root)
    set_parametric_curve_params_to_element(
        function_type_str='0', params=[gamma],
        parameteric_curve_element=parametric_curve_element)

    tree.write(output_name, short_empty_elements=False)


def create_sample_profile():
    tree = ET.parse("./icc_profile_sample/base_profile_v4.xml")
    root = tree.getroot()

    # Profile header
    set_value_to_specific_header_tag(
        root, "PreferredCMMType", "")
    set_value_to_specific_header_tag(
        root, "ProfileVersion", "4.30")
    set_value_to_specific_header_tag(
        root, "CreationDateTime", create_current_date_str())
    set_value_to_specific_header_tag(
        root, "PrimaryPlatform", "MSFT")
    set_value_to_specific_header_tag(
        root, "DeviceManufacturer", "")
    set_value_to_specific_header_tag(
        root, "DeviceModel", "")
    set_value_to_specific_header_tag(
        root, "ProfileCreator", "")
    set_value_to_specific_header_tag(
        root, "ProfileID", "")

    # Tagged element data
    desc_element = get_desc_element(root)
    desc_element.text = "Gamma=3.5, ACES AP0, D65"

    cprt_element = get_cprt_element(root)
    cprt_element.text = "Copyright 2020 Toru Yoshihara"

    chad_mtx = ipcp.calc_chromatic_adaptation_matrix(
        src_white=ipcp.D65, dst_white=ipcp.PCS_D50)
    chad_mtx_element = get_chad_mtx_element(root)
    set_chad_matrix_to_chad_mtx_element(
        mtx=chad_mtx, chad_mtx_element=chad_mtx_element)

    lumi_element = get_lumi_element(root)
    set_lumi_params_to_element(
        luminance=100.0, lumi_element=lumi_element)

    wtpt_element = get_wtpt_element(root)
    set_wtpt_params_to_element(
        wtpt=ipcp.PCS_D50_XYZ, wtpt_element=wtpt_element)

    parametric_curve_element = get_parametric_curve_element(root)
    set_parametric_curve_params_to_element(
        function_type_str='0', params=[2.4],
        parameteric_curve_element=parametric_curve_element)

    rgbXYZ_element_list = get_rgbXYZ_element_list(root)
    src2pcs_mtx = ipcp.calc_rgb_to_xyz_mtx_included_chad_mtx(
        rgb_primaries=cs.get_primaries(cs.P3_D65),
        src_white=ipcp.D65, dst_white=ipcp.PCS_D50)
    set_rgbXYZ_params_to_element(
        src2pcs_mtx=src2pcs_mtx, rgb_XYZ_element_list=rgbXYZ_element_list)

    tree.write("test_out.xml", short_empty_elements=False)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_sample_profile()
    # xml_parse_test()
