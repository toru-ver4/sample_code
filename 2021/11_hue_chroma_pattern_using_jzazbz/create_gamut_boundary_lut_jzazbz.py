# -*- coding: utf-8 -*-
"""
create gamut boundary lut of Jzazbz color space.
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np

# import my libraries
import color_space as cs
from create_gamut_booundary_lut\
    import calc_chroma_boundary_specific_ligheness_jzazbz,\
    create_focal_point_lut_jzazbz
from common import MeasureExecTime

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def make_gb_lut_fname(
        color_space_name, luminance, lightness_num, hue_num):
    fname = f"./lut/JzChz_gb-lut_{color_space_name}_"
    fname += f"{luminance}nits_jj-{lightness_num}_"
    fname += f"hh-{hue_num}.npy"

    return fname


def make_focal_lut_fname(
        luminance, lightness_num, hue_num, prefix="BT709_BT2020"):
    fname = f"./lut/JzChz_focal-lut_{prefix}_"
    fname += f"{luminance}nits_jj-{lightness_num}_"
    fname += f"hh-{hue_num}.npy"

    return fname


def create_jzazbz_gamut_boundary_lut(
        hue_sample=256, lightness_sample=256,
        color_space_name=cs.BT2020, luminance=10000):
    """
    Parameters
    ----------
    hue_sample : int
        The number of hue
    lightness_sample : int
        The number of lightness
    color_space_name : strings
        color space name for colour.RGB_COLOURSPACES
    luminance : float
        peak luminance for Jzazbz color space
    """

    lut = []
    met = MeasureExecTime()
    met.start()
    for j_val in np.linspace(0, 1, lightness_sample):
        print(f"j_val = {j_val:.3f}, ", end="")
        met.lap()
        jzczhz = calc_chroma_boundary_specific_ligheness_jzazbz(
            lightness=j_val, hue_sample=hue_sample,
            cs_name=color_space_name, peak_luminance=luminance)
        lut.append(jzczhz)
    met.end()
    lut = np.array(lut)

    fname = make_gb_lut_fname(
        color_space_name=color_space_name, luminance=luminance,
        lightness_num=lightness_sample, hue_num=hue_sample)
    np.save(fname, lut)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=256, lightness_sample=256,
    #     color_space_name=cs.BT2020, luminance=10000)
    create_jzazbz_gamut_boundary_lut(
        hue_sample=16, lightness_sample=256,
        color_space_name=cs.BT2020, luminance=10000)
    create_jzazbz_gamut_boundary_lut(
        hue_sample=16, lightness_sample=256,
        color_space_name=cs.BT709, luminance=10000)
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=64, lightness_sample=64,
    #     color_space_name=cs.BT2020, luminance=10000)
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=64, lightness_sample=64,
    #     color_space_name=cs.BT709, luminance=10000)
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=4096, lightness_sample=4096,
    #     color_space_name=cs.BT2020, luminance=10000)
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=4096, lightness_sample=4096,
    #     color_space_name=cs.BT709, luminance=10000)
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=4096, lightness_sample=4096,
    #     color_space_name=cs.P3_D65, luminance=10000)

    sample_num = 1024
    luminance = 10000

    inner_lut_name = make_gb_lut_fname(
        color_space_name=cs.BT709, luminance=luminance,
        lightness_num=sample_num, hue_num=sample_num)
    outer_lut_name = make_gb_lut_fname(
        color_space_name=cs.BT2020, luminance=luminance,
        lightness_num=sample_num, hue_num=sample_num)
    inner_lut = np.load(inner_lut_name)
    outer_lut = np.load(outer_lut_name)

    focal_array = create_focal_point_lut_jzazbz(
        inner_lut=inner_lut, outer_lut=outer_lut)
    focal_lut_name = make_focal_lut_fname(
        luminance=luminance, lightness_num=sample_num, hue_num=sample_num,
        prefix="BT709-BT2020")
    np.save(focal_lut_name, focal_array)
