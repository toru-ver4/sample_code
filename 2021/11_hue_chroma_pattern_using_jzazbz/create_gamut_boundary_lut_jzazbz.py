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
        create_jzazbz_gamut_boundary_lut

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=256, lightness_sample=256,
    #     color_space_name=cs.BT2020, luminance=10000)
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=16, lightness_sample=256,
    #     color_space_name=cs.BT2020, luminance=10000)
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=16, lightness_sample=256,
    #     color_space_name=cs.BT709, luminance=10000)
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=64, lightness_sample=64,
    #     color_space_name=cs.BT2020, luminance=10000)
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=64, lightness_sample=64,
    #     color_space_name=cs.BT709, luminance=10000)
    lightness_sample_num = 1024
    hue_sample_num = 4096
    luminance = 10000
    # chroma_sample = 16384
    chroma_sample = 65536
    # create_jzazbz_gamut_boundary_lut_type2(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     chroma_sample=chroma_sample, color_space_name=cs.BT709,
    #     luminance=luminance)
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     color_space_name=cs.BT2020, luminance=luminance)
    # create_jzazbz_gamut_boundary_lut_type2(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     chroma_sample=chroma_sample, color_space_name=cs.P3_D65,
    #     luminance=luminance)

    # luminance = 4000
    # create_jzazbz_gamut_boundary_lut_type2(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     chroma_sample=chroma_sample, color_space_name=cs.BT709,
    #     luminance=luminance)
    # create_jzazbz_gamut_boundary_lut_type2(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     chroma_sample=chroma_sample, color_space_name=cs.BT2020,
    #     luminance=luminance)
    # create_jzazbz_gamut_boundary_lut_type2(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     chroma_sample=chroma_sample, color_space_name=cs.P3_D65,
    #     luminance=luminance)

    luminance = 1000
    # create_jzazbz_gamut_boundary_lut_type2(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     chroma_sample=chroma_sample, color_space_name=cs.BT709,
    #     luminance=luminance)
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     color_space_name=cs.BT2020, luminance=luminance)
    # create_jzazbz_gamut_boundary_lut_type2(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     chroma_sample=chroma_sample, color_space_name=cs.P3_D65,
    #     luminance=luminance)

    luminance = 100
    # create_jzazbz_gamut_boundary_lut_type2(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     chroma_sample=chroma_sample, color_space_name=cs.BT709,
    #     luminance=luminance)
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     color_space_name=cs.BT2020, luminance=luminance)
    # create_jzazbz_gamut_boundary_lut_type2(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     chroma_sample=chroma_sample, color_space_name=cs.P3_D65,
    #     luminance=luminance)

    # luminance = 10000
    # apply_lpf_to_focal_lut(
    #     luminance, lightness_sample_num, hue_sample_num,
    #     prefix="BT709-BT2020",
    #     maximum_l_focal=0.8, minimum_l_focal=0.3)
    luminance = 1000
    apply_lpf_to_focal_lut(
        luminance, lightness_sample_num, hue_sample_num,
        prefix="BT709-BT2020",
        maximum_l_focal=0.33, minimum_l_focal=0.155)
    # luminance = 100
    # apply_lpf_to_focal_lut(
    #     luminance, lightness_sample_num, hue_sample_num,
    #     prefix="BT709-BT2020",
    #     maximum_l_focal=0.132, minimum_l_focal=0.06)

    # lut_name = make_jzazbz_gb_lut_fname(
    #     color_space_name=cs.BT709, luminance=luminance,
    #     lightness_num=lightness_sample_num, hue_num=hue_sample_num)
    # lut = TyLchLut(lut=np.load(lut_name))
