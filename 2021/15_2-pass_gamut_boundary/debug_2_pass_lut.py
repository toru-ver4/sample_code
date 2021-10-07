# -*- coding: utf-8 -*-
"""

```
"""

# import standard libraries
import os
from multiprocessing import Pool, cpu_count

# import third-party libraries
import numpy as np
from colour.utilities.array import tstack
from colour import XYZ_to_RGB, xy_to_XYZ, RGB_COLOURSPACES

# import my libraries
import plot_utility as pu
import color_space as cs
from create_gamut_booundary_lut import CIELAB_CHROMA_MAX, TyLchLut,\
    create_jzazbz_gamut_boundary_lut_type2, is_out_of_gamut_rgb,\
    JZAZBZ_CHROMA_MAX, make_jzazbz_gb_lut_fname_method_c,\
    make_jzazbz_gb_lut_fname_methodb_b,\
    create_cielab_gamut_boundary_lut_method_b,\
    make_cielab_gb_lut_fname_method_b, make_cielab_gb_lut_fname_method_c
from jzazbz import jzazbz_to_large_xyz, jzczhz_to_jzazbz, large_xyz_to_jzazbz
from color_space_plot import create_valid_jzazbz_ab_plane_image_st2084,\
    create_valid_jzazbz_cj_plane_image_st2084

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def calc_chroma_boundary_specific_ligheness_cielab_method_c(
        lch, cs_name, c0):
    """
    parameters
    ----------
    lightness : float
        lightness value(Jzazbz). range is 0.0 - 1.0.
    hue_sample : int
        Sample number of the Hue
    cs_name : string
        A color space name. ex. "ITU-R BT.709", "ITU-R BT.2020"

    Examples
    --------
    >>> boundary_jch = calc_chroma_boundary_specific_ligheness_jzazbz(
    ...     lightness=0.5, hue_sample=16, cs_name=cs.BT2020,
    ...     peak_luminance=10000)
    [[  5.00000000e-01   2.72627831e-01   0.00000000e+00]
     [  5.00000000e-01   2.96944618e-01   2.40000000e+01]
     [  5.00000000e-01   3.19167137e-01   4.80000000e+01]
     [  5.00000000e-01   2.51322746e-01   7.20000000e+01]
     [  5.00000000e-01   2.41002083e-01   9.60000000e+01]
     [  5.00000000e-01   2.76854515e-01   1.20000000e+02]
     [  5.00000000e-01   3.99024010e-01   1.44000000e+02]
     [  5.00000000e-01   2.64456749e-01   1.68000000e+02]
     [  5.00000000e-01   2.32390404e-01   1.92000000e+02]
     [  5.00000000e-01   2.51740456e-01   2.16000000e+02]
     [  5.00000000e-01   3.38995934e-01   2.40000000e+02]
     [  5.00000000e-01   3.09918404e-01   2.64000000e+02]
     [  5.00000000e-01   2.71250725e-01   2.88000000e+02]
     [  5.00000000e-01   2.59991646e-01   3.12000000e+02]
     [  5.00000000e-01   2.63157845e-01   3.36000000e+02]
     [  5.00000000e-01   2.72627831e-01   3.60000000e+02]]
    """
    # lch --> rgb

    ll = lch[..., 0]
    chroma_init = lch[..., 1]
    hue = np.deg2rad(lch[..., 2])

    trial_num = 20

    r_val = chroma_init

    for t_idx in range(trial_num):
        aa = r_val * np.cos(hue)
        bb = r_val * np.sin(hue)
        lab = tstack((ll, aa, bb))
        rgb = cs.lab_to_rgb(lab, cs_name)

        ng_idx = is_out_of_gamut_rgb(rgb=rgb)
        ok_idx = np.logical_not(ng_idx)
        add_sub = c0 / (2 ** (t_idx))
        r_val[ok_idx] = r_val[ok_idx] + add_sub
        r_val[~ok_idx] = r_val[~ok_idx] - add_sub

    zero_idx = (chroma_init <= 0)
    r_val[zero_idx] = 0.0

    lch_result = tstack([ll, r_val, np.rad2deg(hue)])

    return lch_result


def plot_d65_multi_luminance():
    range = 0.0003
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="D65 in the az-bz plane",
        graph_title_size=None,
        xlabel="az", ylabel="bz",
        axis_label_size=None,
        legend_size=17,
        xlim=[-range, range],
        ylim=[-range, range],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)

    luminance_list = [0, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    for luminance in luminance_list:
        d65_xyz = xy_to_XYZ(cs.D65) * luminance
        jzazbz = large_xyz_to_jzazbz(d65_xyz)
        az = jzazbz[..., 1]
        bz = jzazbz[..., 2]
        ax1.plot(az, bz, 'o', label=f"{luminance} nits")

    fname = "./img/white_posi.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc='upper right', show=False, save_fname=fname)


def create_lab_gamut_boundary_method_c(
        hue_sample=8, lightness_sample=8, chroma_sample=1024,
        color_space_name=cs.BT709, luminance=100):

    ll_num = lightness_sample
    lut = create_cielab_gamut_boundary_lut_method_b(
        lightness_sample=ll_num, chroma_sample=chroma_sample,
        hue_sample=hue_sample, cs_name=color_space_name)
    np.save(make_cielab_gb_lut_fname_method_b(
        color_space_name=color_space_name, lightness_num=lightness_sample,
        hue_num=hue_sample), lut)

    # create 2d lut using method B
    lut_b = np.load(
        make_cielab_gb_lut_fname_method_b(
            color_space_name=color_space_name, lightness_num=lightness_sample,
            hue_num=hue_sample))

    # create 2d lut using method C
    c0 = CIELAB_CHROMA_MAX / (chroma_sample - 1)
    lut_c = np.zeros_like(lut_b)
    for l_idx in range(lightness_sample):
        jzczhz_init = lut_b[l_idx]
        jzczhz = calc_chroma_boundary_specific_ligheness_cielab_method_c(
            lch=jzczhz_init, cs_name=color_space_name,
            peak_luminance=luminance, c0=c0)
        lut_c[l_idx] = jzczhz

    fname = make_cielab_gb_lut_fname_method_c(
        color_space_name=color_space_name,
        lightness_num=lightness_sample, hue_num=hue_sample)
    np.save(fname, np.float32(lut_c))


def debug_plot_cielab():
    chroma_sample = 256
    hue_sample = 1024
    lightness_sample = 1024
    luminance = 100
    h_num_intp = 128
    l_num_intp = 128
    color_space_name = cs.BT709
    create_lab_gamut_boundary_method_c(
        hue_sample=hue_sample, lightness_sample=lightness_sample,
        chroma_sample=chroma_sample,
        color_space_name=color_space_name, luminance=luminance)
    debug_plot_cielab_ab_plane_with_interpolation(
        hue_sample=hue_sample, lightness_sample=lightness_sample,
        j_num_intp=j_num_intp,
        color_space_name=color_space_name, luminance=luminance)
    plot_cielab_cl_plane_with_interpolation(
        hue_sample=hue_sample, lightness_sample=lightness_sample,
        h_num_intp=h_num_intp,
        color_space_name=color_space_name, luminance=luminance)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))