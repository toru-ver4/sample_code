# -*- coding: utf-8 -*-
"""
パラメータ置き場
================

"""

# import standard libraries
import os

# import third-party libraries

# import my libraries
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


L_SAMPLE_NUM_MAX = 8192
H_SAMPLE_NUM_MAX = 8192

GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE = 1024
GAMUT_BOUNDARY_LUT_HUE_SAMPLE = 1024
CHROMA_MAP_DEGREE_SAMPLE_NUM = 1024

DIPS_150_SAMPLE_ST_BT2020 = int(115/360 * GAMUT_BOUNDARY_LUT_HUE_SAMPLE)
DIPS_150_SAMPLE_ED_BT2020 = int(160/360 * GAMUT_BOUNDARY_LUT_HUE_SAMPLE)
DIPS_300_SAMPLE_ST_BT2020 = int(270/360 * GAMUT_BOUNDARY_LUT_HUE_SAMPLE)
DIPS_300_SAMPLE_ED_BT2020 = int(324/360 * GAMUT_BOUNDARY_LUT_HUE_SAMPLE)

DIPS_150_SAMPLE_ST_P3 = int(115/360 * GAMUT_BOUNDARY_LUT_HUE_SAMPLE)
DIPS_150_SAMPLE_ED_P3 = int(160/360 * GAMUT_BOUNDARY_LUT_HUE_SAMPLE)
DIPS_300_SAMPLE_ST_P3 = int(325/360 * GAMUT_BOUNDARY_LUT_HUE_SAMPLE)
DIPS_300_SAMPLE_ED_P3 = int(359/360 * GAMUT_BOUNDARY_LUT_HUE_SAMPLE)

# BT.2407 の FIGURE A2-4 を見ると 240° くらいで終わってるので…
L_FOCAL_240_INDEX_BT2020 = int(240/360 * GAMUT_BOUNDARY_LUT_HUE_SAMPLE)
L_FOCAL_240_INDEX_P3 = int(225/360 * GAMUT_BOUNDARY_LUT_HUE_SAMPLE)

C_FOCAL_MAX_VALUE = 5000
LPF_WN_PARAM = 0.4 * (256/GAMUT_BOUNDARY_LUT_HUE_SAMPLE)
LPF_NN_PARAM = int(4 * (GAMUT_BOUNDARY_LUT_HUE_SAMPLE/256) + 0.5)


def get_gamut_boundary_lut_name(
        color_space_name=cs.BT709,
        luminance_sample_num=GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE,
        hue_sample_num=GAMUT_BOUNDARY_LUT_HUE_SAMPLE):
    name = f"./luts/GamutBoundaryLUT_{color_space_name}_"\
        + f"L_{luminance_sample_num}_H_{hue_sample_num}.npy"
    return name


def get_chroma_map_lut_name(
        outer_color_space_name=cs.BT2020,
        inner_color_space_name=cs.BT709,
        luminance_sample_num=GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE,
        hue_sample_num=GAMUT_BOUNDARY_LUT_HUE_SAMPLE,
        focal_type="Lfocal"):
    name = f"./luts/ChromaMap{focal_type}LUT_{outer_color_space_name}_to_"\
        + f"{inner_color_space_name}_"\
        + f"L_{luminance_sample_num}_H_{hue_sample_num}.npy"
    return name


def get_l_cusp_name(
        outer_color_space_name=cs.BT2020,
        inner_color_space_name=cs.BT709,
        luminance_sample_num=GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE,
        hue_sample_num=GAMUT_BOUNDARY_LUT_HUE_SAMPLE):
    name = f"./luts/LCuspLUT_{outer_color_space_name}_to_"\
        + f"{inner_color_space_name}_"\
        + f"L_{luminance_sample_num}_H_{hue_sample_num}.npy"
    return name


def get_focal_name(
        outer_color_space_name=cs.BT2020,
        inner_color_space_name=cs.BT709,
        luminance_sample_num=GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE,
        hue_sample_num=GAMUT_BOUNDARY_LUT_HUE_SAMPLE,
        focal_type="Lfocal"):
    name = f"./luts/{focal_type}LUT_{outer_color_space_name}_to_"\
        + f"{inner_color_space_name}_"\
        + f"L_{luminance_sample_num}_H_{hue_sample_num}.npy"
    return name


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(get_gamut_boundary_lut_name(cs.BT709, 10, 20))
