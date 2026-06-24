# -*- coding: utf-8 -*-
"""
debug code
==========

"""

# import standard libraries
import os
import sys
import xml.etree.ElementTree as ET
import subprocess
from pathlib import Path
from itertools import product

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TY_LIB_ROOT = PROJECT_ROOT / "ty_lib"
sys.path.append(str(TY_LIB_ROOT))

# import third-party libraries
import numpy as np
from colour.utilities import tstack

# import my libraries
import icc_profile_xml_control as ipxc
import icc_profile_calc_param as ipcp
import color_space as cs
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def make_gamma_lut(gamma=2.4, num_of_entries=1024):
    """
    Create a uint16 TRC LUT for curveType.
    """
    x = np.linspace(0.0, 1.0, num_of_entries)
    y = x ** gamma
    lut = np.uint16(np.round(y * 0xFFFF))

    return lut


def create_bt709_gamma24_curve_1024_profile():
    """
    Create BT.709/D65 ICC profile XML with Gamma 2.4 curveType TRC.
    """
    template_fname = TY_LIB_ROOT / "icc_profile_sample" / "base_profile_v4.xml"
    output_fname = Path("bt709_gamma24_curve_1024.xml")

    tree = ET.parse(template_fname)
    root = tree.getroot()

    ipxc.create_profle_header(root)

    desc_element = ipxc.get_desc_element(root)
    desc_element.text = "BT.709 Gamma 2.4 CurveType 1024 entries"

    cprt_element = ipxc.get_cprt_element(root)
    cprt_element.text = "Copyright 2026 Toru Yoshihara"

    src_white = cs.D65
    dst_white = ipcp.PCS_D50
    src_primaries = cs.get_primaries(cs.BT709)

    chad_mtx = ipcp.calc_chromatic_adaptation_matrix(
        src_white=src_white, dst_white=dst_white)
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
        src_white=src_white, dst_white=dst_white)
    ipxc.set_rgbXYZ_params_to_element(
        src2pcs_mtx=src2pcs_mtx, rgb_XYZ_element_list=rgbXYZ_element_list)

    curve_element = ipxc.create_curve_type_element(root)
    gamma_lut = make_gamma_lut(gamma=2.4, num_of_entries=1024)
    ipxc.set_curve_type_params_to_element(
        lut=gamma_lut, curve_element=curve_element)

    tree.write(output_fname, short_empty_elements=False)

    return output_fname


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_bt709_gamma24_curve_1024_profile()
