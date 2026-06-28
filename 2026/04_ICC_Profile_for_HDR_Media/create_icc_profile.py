# -*- coding: utf-8 -*-
"""
debug code
==========

"""

# import standard libraries
import os
import sys
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


def make_pq_lut(num_of_entries=4096):
    x = np.linspace(0, 1, num_of_entries)
    y = tf.eotf(x, tf.ST2084)
    lut = np.uint16(np.round(y * 0xFFFF))

    return lut


def create_bt709_gamma24_curve_1024_profile():
    """
    Create BT.709/D65 ICC profile XML with Gamma 2.4 curveType TRC.
    """
    xml_fname = Path("./xml/bt709_gamma24_curve_1024.xml")
    icc_fname = Path("./icc/bt709_gamma24_curve_1024.icc")

    tree = ipxc.create_profile_xml()
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

    tree.write(xml_fname, short_empty_elements=False)

    command = "iccFromXml"
    print(f"{command} {xml_fname} {icc_fname}")
    subprocess.run([command, xml_fname, icc_fname])

    return xml_fname


def create_bt2020_pq_curve_4096_with_cicp_profile(
        cicp: list = [9, 16, 9, 1]):
    """
    Create a BT.2020/D65 ICC profile with a PQ curveType TRC and CICP tag.
    """
    xml_fname = Path("./xml/bt2020_PQ_with_CICP.xml")
    icc_fname = Path("./icc/bt2020_PQ_with_CICP.icc")

    tree = ipxc.create_profile_xml()
    root = tree.getroot()

    ipxc.create_profle_header(root)

    desc_element = ipxc.get_desc_element(root)
    desc_element.text = "BT.2020 PQ CurveType 4096 entries with CICP"

    cprt_element = ipxc.get_cprt_element(root)
    cprt_element.text = "Copyright 2026 Toru Yoshihara"

    src_white = cs.D65
    dst_white = ipcp.PCS_D50
    src_primaries = cs.get_primaries(cs.BT2020)

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
    gamma_lut = make_pq_lut(num_of_entries=4096)
    ipxc.set_curve_type_params_to_element(
        lut=gamma_lut, curve_element=curve_element)

    ipxc.create_cicp_tag(root, cicp)

    tree.write(xml_fname, short_empty_elements=False)

    command = "iccFromXml"
    print(f"{command} {xml_fname} {icc_fname}")
    subprocess.run([command, xml_fname, icc_fname], check=True)

    return xml_fname


def create_mhc_icc_profile(
        gamma=2.4, src_white=np.array([0.3127, 0.3290]),
        src_primaries=np.array([[0.680, 0.320], [0.265, 0.690], [0.15, 0.06]]),
        desc_str="Gamam=2.4_DCI-P3_D65",
        cprt_str="Copyright 2020 HOGEHOGE Corp.",
        xml_fname="Gamam=2.4_DCI-P3_D65.xml",
        icc_fname="Gamam=2.4_DCI-P3_D65.icm",
        min_luminance=0.005,
        peak_luminance=1000,
        max_full_frame_luminance=300,
        calibration_matrix=np.identity(3),
        calibration_luts=np.zeros((256, 3))):
    """
    create simple profile.
    gamma function must be "y = x ** gamma" format.
    """
    tree = ipxc.create_profile_xml(include_mhc2=True)
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
        luminance=max_full_frame_luminance, lumi_element=lumi_element)

    wtpt_element = ipxc.get_wtpt_element(root)
    ipxc.set_wtpt_params_to_element(
        wtpt=ipcp.PCS_D50_XYZ, wtpt_element=wtpt_element)

    rgbXYZ_element_list = ipxc.get_rgbXYZ_element_list(root)
    src2pcs_mtx = ipcp.calc_rgb_to_xyz_mtx_included_chad_mtx(
        rgb_primaries=src_primaries,
        src_white=src_white, dst_white=ipcp.D65)
    ipxc.set_rgbXYZ_params_to_element(
        src2pcs_mtx=src2pcs_mtx, rgb_XYZ_element_list=rgbXYZ_element_list)

    parametric_curve_element = ipxc.get_parametric_curve_element(root)
    ipxc.set_parametric_curve_params_to_element(
        function_type_str='0', params=[gamma],
        parameteric_curve_element=parametric_curve_element)

    mhc2_element = ipxc.get_mhc2_element(root)
    ipxc.set_mhc2_params_to_element(
        mhc2_element=mhc2_element,
        min_luminance=min_luminance, peak_luminance=peak_luminance,
        matrix=calibration_matrix, luts=calibration_luts)

    tree.write(xml_fname, short_empty_elements=False)
    command = "iccFromXml"
    print(f"{command} {xml_fname} {icc_fname}")
    subprocess.run([command, xml_fname, icc_fname])


def create_mhc2_profile_with_gain(
        gain=0.5,
        min_luminance=0.001,
        peak_luminance=450,
        max_full_frame_luminance=250,
        cs_name=cs.BT2020):
    calibration_matrix = np.identity(3)
    luminance_str = f"{min_luminance}-{peak_luminance}-"
    # luminance_str += f"{peak_luminance}-"
    luminance_str += f"{max_full_frame_luminance}"
    luts = ipcp.create_gain_1dlut_for_st2084(num_of_sample=1024, gain=gain)
    xml_fname = "./xml/MHC2_sample.xml"
    cs_name_file = cs_name.replace(" ", "_")
    icc_fname = f"./icc/MHC2_{luminance_str}-nits_gain-{gain:.3f}_"
    icc_fname += f"{cs_name_file}.icm"
    create_mhc_icc_profile(
        gamma=2.4, src_white=cs.D65,
        src_primaries=cs.get_primaries(cs_name),
        desc_str=str(Path(icc_fname).stem),
        cprt_str="Copyright 2024 Toru Yoshihara",
        min_luminance=min_luminance,
        peak_luminance=peak_luminance,
        max_full_frame_luminance=max_full_frame_luminance,
        xml_fname=xml_fname,
        icc_fname=icc_fname,
        calibration_luts=luts,
        calibration_matrix=calibration_matrix)


def create_gamma24_bt2020():
    xml_fname = "./xml/Gamma2.4_BT.2020_D65_ty.xml"
    icc_fname = "./icc/Gamma2.4_BT.2020_D65.icc"
    ipxc.create_simple_power_gamma_profile(
        gamma=2.4, src_white=cs.D65,
        src_primaries=cs.get_primaries(cs.BT2020),
        desc_str="Gamma2.4_BT.2020_D65",
        cprt_str="Copyright 2020 Toru Yoshihara.",
        output_name=xml_fname
    )

    command = "iccFromXml"
    print(f"{command} {xml_fname} {icc_fname}")
    subprocess.run([command, xml_fname, icc_fname])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_gamma24_bt2020()
    # create_mhc2_profile_with_gain(
    #     gain=0.5,
    #     min_luminance=0.1,
    #     peak_luminance=700,
    #     max_full_frame_luminance=700,
    #     cs_name=cs.BT2020
    # )
    create_bt2020_pq_curve_4096_with_cicp_profile(cicp=[9, 16, 9, 1])
