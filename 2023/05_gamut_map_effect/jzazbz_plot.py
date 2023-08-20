# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from pathlib import Path

# import third-party libraries
import numpy as np

# import my libraries
import plot_utility as pu
import color_space as cs
from create_gamut_booundary_lut import create_jzazbz_gamut_boundary_lut_type2,\
    make_jzazbz_gb_lut_fname_methodb_b, make_jzazbz_gb_lut_fname_method_c,\
    calc_chroma_boundary_specific_ligheness_jzazbz_method_c,\
    JZAZBZ_CHROMA_MAX
from jzazbz_azbz_czhz_plot import plot_cj_plane_with_interpolation

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2023 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def create_jzazbz_gamut_boundary_method_c(
        hue_sample=8, lightness_sample=8, chroma_sample=1024,
        color_space_name=cs.BT709, luminance=100):

    c0 = JZAZBZ_CHROMA_MAX / (chroma_sample - 1)

    method_b_lut_name = make_jzazbz_gb_lut_fname_methodb_b(
        color_space_name=color_space_name, luminance=luminance,
        lightness_num=lightness_sample, hue_num=hue_sample)

    method_c_lut_name = make_jzazbz_gb_lut_fname_method_c(
        color_space_name=color_space_name, luminance=luminance,
        lightness_num=lightness_sample, hue_num=hue_sample)

    # create 2d lut using method B

    if Path(method_b_lut_name).is_file():
        print(f"{method_b_lut_name} is exist!!")
    else:
        print(f"{method_b_lut_name} is not exist... create the new one")
        create_jzazbz_gamut_boundary_lut_type2(
            hue_sample=hue_sample, lightness_sample=lightness_sample,
            chroma_sample=chroma_sample, color_space_name=color_space_name,
            luminance=luminance)

    lut_b = np.load(method_b_lut_name)

    if Path(method_c_lut_name).is_file():
        print(f"{method_c_lut_name} is exist!!")
    else:
        # create 2d lut using method C
        print(f"{method_c_lut_name} is not exist... create the new one")
        lut_c = np.zeros_like(lut_b)
        for l_idx in range(lightness_sample):
            jzczhz_init = lut_b[l_idx]
            jzczhz = calc_chroma_boundary_specific_ligheness_jzazbz_method_c(
                lch=jzczhz_init, cs_name=color_space_name,
                peak_luminance=luminance, c0=c0)
            lut_c[l_idx] = jzczhz

        np.save(method_c_lut_name, np.float32(lut_c))


def create_jzazbz_2dlut_using_method_c(
        luminance=1000, hue_num=4096, lightness_sample=1024,
        chroma_sample=512, color_space_name=cs.BT709):
    create_jzazbz_gamut_boundary_method_c(
        hue_sample=hue_num, lightness_sample=lightness_sample,
        chroma_sample=chroma_sample, color_space_name=color_space_name,
        luminance=luminance)


def debug_plot_jzazbz(
        hue_num, lightness_sample, luminance, h_num_intp, j_num_intp,
        color_space_name):
    plot_cj_plane_with_interpolation(
        hue_sample=hue_num, lightness_sample=lightness_sample,
        h_num_intp=h_num_intp, color_space_name=color_space_name,
        luminance=luminance)


def create_jzazbz_cl_plane_and_plot():
    hue_sample = 4096
    hue_sample_intp = 361
    lightness_sample = 1024
    chroma_sample = 1024
    color_space_name = cs.BT2020
    luminance = 100
    create_jzazbz_gamut_boundary_method_c(
        hue_sample=hue_sample, lightness_sample=lightness_sample,
        chroma_sample=chroma_sample, color_space_name=color_space_name,
        luminance=luminance)
    plot_cj_plane_with_interpolation(
        hue_sample=hue_sample, lightness_sample=lightness_sample,
        h_num_intp=hue_sample_intp, color_space_name=color_space_name,
        luminance=luminance)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_jzazbz_cl_plane_and_plot()
