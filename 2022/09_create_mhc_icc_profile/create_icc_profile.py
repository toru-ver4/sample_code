# -*- coding: utf-8 -*-
"""
debug code
==========

"""

# import standard libraries
import os
import xml.etree.ElementTree as ET
import subprocess

# import third-party libraries
import numpy as np
from colour.utilities import tstack

# import my libraries
import icc_profile_xml_control as ipxc
import icc_profile_calc_param as ipcp
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def main_func():
    pass


def create_mhc_icc_profile(
        gamma=2.4, src_white=np.array([0.3127, 0.3290]),
        src_primaries=np.array([[0.680, 0.320], [0.265, 0.690], [0.15, 0.06]]),
        desc_str="Gamam=2.4_DCI-P3_D65",
        cprt_str="Copyright 2020 HOGEHOGE Corp.",
        xml_fname="Gamam=2.4_DCI-P3_D65.xml",
        icc_fname="Gamam=2.4_DCI-P3_D65.icm",
        calibration_matrix=np.identity(3),
        calibration_luts=np.zeros((256, 3))):
    """
    create simple profile.
    gamma function must be "y = x ** gamma" format.
    """
    tree = ET.parse("./xml/base_profile_v4_mhc.xml")
    root = tree.getroot()

    # Profile header
    ipxc.create_profle_header(root)

    # Tagged element data
    desc_element = ipxc.get_desc_element(root)
    desc_element.text = desc_str

    cprt_element = ipxc.get_cprt_element(root)
    cprt_element.text = cprt_str

    chad_mtx = ipcp.calc_chromatic_adaptation_matrix(
        src_white=src_white, dst_white=ipcp.PCS_D50)
    chad_mtx_element = ipxc.get_chad_mtx_element(root)
    ipxc.set_chad_matrix_to_chad_mtx_element(
        mtx=chad_mtx, chad_mtx_element=chad_mtx_element)

    lumi_element = ipxc.get_lumi_element(root)
    ipxc.set_lumi_params_to_element(
        luminance=100.0, lumi_element=lumi_element)

    wtpt_element = ipxc.get_wtpt_element(root)
    ipxc.set_wtpt_params_to_element(
        wtpt=ipcp.PCS_D50_XYZ, wtpt_element=wtpt_element)

    rgbXYZ_element_list = ipxc.get_rgbXYZ_element_list(root)
    src2pcs_mtx = ipcp.calc_rgb_to_xyz_mtx_included_chad_mtx(
        rgb_primaries=src_primaries,
        src_white=src_white, dst_white=ipcp.PCS_D50)
    ipxc.set_rgbXYZ_params_to_element(
        src2pcs_mtx=src2pcs_mtx, rgb_XYZ_element_list=rgbXYZ_element_list)

    parametric_curve_element = ipxc.get_parametric_curve_element(root)
    ipxc.set_parametric_curve_params_to_element(
        function_type_str='0', params=[gamma],
        parameteric_curve_element=parametric_curve_element)

    mhc2_element = ipxc.get_mhc2_element(root)
    ipxc.set_mhc2_params_to_element(
        mhc2_element=mhc2_element, min_luminance=0.1, peak_luminance=600,
        matrix=calibration_matrix, luts=calibration_luts)

    tree.write(xml_fname, short_empty_elements=False)
    subprocess.run(["iccFromXml", xml_fname, icc_fname])


def parse_mhc2_data():
    tree = ET.parse("./xml/base_profile_v4_mhc.xml")
    root = tree.getroot()
    mhc_tag_element = ipxc.get_icc_tag_element(
        root, parent_tag="PrivateType",
        key_tag="TagSignature", key_text="MHC2", target_tag="UnknownData")

    raw_data = mhc_tag_element.text
    for ii in range(10):
        print(f"{ii}, {raw_data[ii]}")


def create_sample_identity_1dlut(num_of_sample):
    x = np.linspace(0, 1, num_of_sample)
    y = x

    lut = tstack([y, y, y])

    return lut


def debug_func():
    # parse_mhc2_data()
    luts = create_sample_identity_1dlut(256)
    xml_fname = "./xml/sRGB_MHC2_sample.xml"
    icc_fname = "./icc_profile/GM24_BT709_MHC2_sample.icm"
    create_mhc_icc_profile(
        gamma=2.4, src_white=cs.D65,
        src_primaries=cs.get_primaries(cs.BT709),
        desc_str="sRGB_MHC2_Sample,",
        cprt_str="Copyright 2022 Toru Yoshihara",
        xml_fname=xml_fname,
        calibration_luts=luts)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # main_func()
    debug_func()
