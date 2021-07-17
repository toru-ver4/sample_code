# -*- coding: utf-8 -*-
"""
create gamut boundary lut of Jzazbz color space.
"""

# import standard libraries
import os
import ctypes
from colour.utilities.array import tstack

# import third-party libraries
import numpy as np
from multiprocessing import Pool, cpu_count, Array

from colour import xy_to_XYZ

# import my libraries
import color_space as cs
from create_gamut_booundary_lut\
    import create_jzazbz_gamut_boundary_lut_type2,\
        apply_lpf_to_focal_lut, make_jzazbz_gb_lut_fname,\
        create_jzazbz_gamut_boundary_lut,\
        create_jzazbz_gamut_boundary_lut_type3

from create_gamut_booundary_lut import shm, shm_buf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def create_gamut_boundary():
    lightness_sample_num = 1024
    hue_sample_num = 4096
    luminance = 10000
    chroma_sample = 2 ** 16

    # create_jzazbz_gamut_boundary_lut_type3(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     chroma_sample=chroma_sample, color_space_name=cs.BT709,
    #     luminance=luminance)

    # create_jzazbz_gamut_boundary_lut_type2(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     chroma_sample=chroma_sample, color_space_name=cs.BT709,
    #     luminance=luminance)
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     color_space_name=cs.BT2020, luminance=luminance)
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     color_space_name=cs.P3_D65, luminance=luminance)

    luminance = 1000
    # create_jzazbz_gamut_boundary_lut_type2(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     chroma_sample=chroma_sample, color_space_name=cs.BT709,
    #     luminance=luminance)
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     color_space_name=cs.BT2020, luminance=luminance)
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     color_space_name=cs.P3_D65, luminance=luminance)

    luminance = 100
    # create_jzazbz_gamut_boundary_lut_type2(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     chroma_sample=chroma_sample, color_space_name=cs.BT709,
    #     luminance=luminance)
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     color_space_name=cs.BT2020, luminance=luminance)
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     color_space_name=cs.P3_D65, luminance=luminance)


def create_focal_lut():
    lightness_sample_num = 1024
    hue_sample_num = 4096
    luminance = 10000
    apply_lpf_to_focal_lut(
        luminance, lightness_sample_num, hue_sample_num,
        prefix="BT709-BT2020",
        maximum_l_focal=0.8, minimum_l_focal=0.5, wn=0.06)
    luminance = 1000
    apply_lpf_to_focal_lut(
        luminance, lightness_sample_num, hue_sample_num,
        prefix="BT709-BT2020",
        maximum_l_focal=0.33, minimum_l_focal=0.173, wn=0.06)
    luminance = 100
    apply_lpf_to_focal_lut(
        luminance, lightness_sample_num, hue_sample_num,
        prefix="BT709-BT2020",
        maximum_l_focal=0.132, minimum_l_focal=0.068, wn=0.06)

    luminance = 10000
    apply_lpf_to_focal_lut(
        luminance, lightness_sample_num, hue_sample_num,
        prefix="BT709-P3D65",
        maximum_l_focal=0.81, minimum_l_focal=0.4, wn=0.1,
        inner_cs_name=cs.BT709, outer_cs_name=cs.P3_D65)
    luminance = 1000
    apply_lpf_to_focal_lut(
        luminance, lightness_sample_num, hue_sample_num,
        prefix="BT709-P3D65",
        maximum_l_focal=0.34, minimum_l_focal=0.18, wn=0.1,
        inner_cs_name=cs.BT709, outer_cs_name=cs.P3_D65)
    luminance = 100
    apply_lpf_to_focal_lut(
        luminance, lightness_sample_num, hue_sample_num,
        prefix="BT709-P3D65",
        maximum_l_focal=0.13, minimum_l_focal=0.07, wn=0.1,
        inner_cs_name=cs.BT709, outer_cs_name=cs.P3_D65)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_gamut_boundary()
    create_focal_lut()

    del shm_buf
    shm.close()
    shm.unlink()
