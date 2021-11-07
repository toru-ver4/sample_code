# -*- coding: utf-8 -*-
"""

```
"""

# import standard libraries
import os
from itertools import product

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
from jzazbz import large_xyz_to_jzazbz, jzazbz_to_large_xyz, jzczhz_to_jzazbz
from jzazbz_azbz_czhz_plot import debug_plot_jzazbz,\
    plot_cj_plane_with_interpolation_core
from cielab_ab_cl_plot import debug_plot_cielab
from common import MeasureExecTime

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
        color_space_name=cs.BT709):

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
        lch_init = lut_b[l_idx]
        lch_result = calc_chroma_boundary_specific_ligheness_cielab_method_c(
            lch=lch_init, cs_name=color_space_name, c0=c0)
        lut_c[l_idx] = lch_result

    fname = make_cielab_gb_lut_fname_method_c(
        color_space_name=color_space_name,
        lightness_num=lightness_sample, hue_num=hue_sample)
    np.save(fname, np.float32(lut_c))


def calc_chroma_boundary_specific_ligheness_jzazbz_method_c(
        lch, cs_name, peak_luminance, c0):
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

    jj = lch[..., 0]
    chroma_init = lch[..., 1]
    hue = np.deg2rad(lch[..., 2])

    trial_num = 30

    r_val = chroma_init

    for t_idx in range(trial_num):
        aa = r_val * np.cos(hue)
        bb = r_val * np.sin(hue)
        jzazbz = tstack((jj, aa, bb))
        large_xyz = jzazbz_to_large_xyz(jzazbz)
        rgb_luminance = XYZ_to_RGB(
            large_xyz, cs.D65, cs.D65,
            RGB_COLOURSPACES[cs_name].matrix_XYZ_to_RGB)

        ng_idx = is_out_of_gamut_rgb(rgb=rgb_luminance/peak_luminance)
        ok_idx = np.logical_not(ng_idx)
        add_sub = c0 / (2 ** (t_idx))
        r_val[ok_idx] = r_val[ok_idx] + add_sub
        r_val[~ok_idx] = r_val[~ok_idx] - add_sub

    zero_idx = (chroma_init <= 0)
    r_val[zero_idx] = 0.0

    jzczhz = tstack([jj, r_val, np.rad2deg(hue)])

    return jzczhz


def create_jzazbz_gamut_boundary_method_c(
        hue_sample=8, lightness_sample=8, chroma_sample=1024,
        color_space_name=cs.BT709, luminance=100):

    c0 = JZAZBZ_CHROMA_MAX / (chroma_sample - 1)
    # create 2d lut using method B
    create_jzazbz_gamut_boundary_lut_type2(
        hue_sample=hue_sample, lightness_sample=lightness_sample,
        chroma_sample=chroma_sample, color_space_name=color_space_name,
        luminance=luminance)

    lut_b = np.load(
        make_jzazbz_gb_lut_fname_methodb_b(
            color_space_name=color_space_name, luminance=luminance,
            lightness_num=lightness_sample, hue_num=hue_sample))

    # create 2d lut using method C
    lut_c = np.zeros_like(lut_b)
    for l_idx in range(lightness_sample):
        jzczhz_init = lut_b[l_idx]
        jzczhz = calc_chroma_boundary_specific_ligheness_jzazbz_method_c(
            lch=jzczhz_init, cs_name=color_space_name,
            peak_luminance=luminance, c0=c0)
        lut_c[l_idx] = jzczhz

    fname = make_jzazbz_gb_lut_fname_method_c(
        color_space_name=color_space_name, luminance=luminance,
        lightness_num=lightness_sample, hue_num=hue_sample)
    np.save(fname, np.float32(lut_c))


def create_jzazbz_2dlut_using_method_c_and_plot(
        luminance=1000, color_space_name=cs.BT709):
    hue_num = 4096
    lightness_sample = 1024
    chroma_sample = 512
    h_num_intp = 1200
    j_num_intp = 1200
    # create_jzazbz_gamut_boundary_method_c(
    #     hue_sample=hue_num, lightness_sample=lightness_sample,
    #     chroma_sample=chroma_sample, color_space_name=color_space_name,
    #     luminance=luminance)
    debug_plot_jzazbz(
        hue_sample=hue_num, lightness_sample=lightness_sample,
        luminance=luminance, h_num_intp=h_num_intp, j_num_intp=j_num_intp,
        color_space_name=color_space_name)


def create_cielab_2dlut_using_method_c_and_plot(color_space_name=cs.BT709):
    chroma_sample = 512
    hue_sample = 4096
    lightness_sample = 1024
    h_num_intp = 1200
    l_num_intp = 1200

    # create_lab_gamut_boundary_method_c(
    #     hue_sample=hue_sample, lightness_sample=lightness_sample,
    #     chroma_sample=chroma_sample,
    #     color_space_name=color_space_name)
    debug_plot_cielab(
        hue_sample=hue_sample, lightness_sample=lightness_sample,
        h_num_intp=h_num_intp, l_num_intp=l_num_intp,
        color_space_name=color_space_name)


def plot_plane_festival():
    # create_cielab_2dlut_using_method_c_and_plot(color_space_name=cs.BT709)
    # create_cielab_2dlut_using_method_c_and_plot(color_space_name=cs.P3_D65)
    # create_cielab_2dlut_using_method_c_and_plot(color_space_name=cs.BT2020)

    # create_jzazbz_2dlut_using_method_c_and_plot(
    #     luminance=100, color_space_name=cs.BT2020)
    # create_jzazbz_2dlut_using_method_c_and_plot(
    #     luminance=1000, color_space_name=cs.BT2020)
    # create_jzazbz_2dlut_using_method_c_and_plot(
    #     luminance=10000, color_space_name=cs.BT2020)
    # create_jzazbz_2dlut_using_method_c_and_plot(
    #     luminance=100, color_space_name=cs.BT709)
    # create_jzazbz_2dlut_using_method_c_and_plot(
    #     luminance=1000, color_space_name=cs.BT709)
    # create_jzazbz_2dlut_using_method_c_and_plot(
    #     luminance=10000, color_space_name=cs.BT709)
    # create_jzazbz_2dlut_using_method_c_and_plot(
    #     luminance=100, color_space_name=cs.P3_D65)
    # create_jzazbz_2dlut_using_method_c_and_plot(
    #     luminance=1000, color_space_name=cs.P3_D65)
    # create_jzazbz_2dlut_using_method_c_and_plot(
    #     luminance=10000, color_space_name=cs.P3_D65)


def debug_ng_cusp():
    bg_lut_name = make_jzazbz_gb_lut_fname_method_c(
        color_space_name=cs.BT709, luminance=1000)
    bg_lut = TyLchLut(lut=np.load(bg_lut_name))
    hue_list = np.linspace(250, 260, 256)
    for hue in hue_list:
        cusp = bg_lut.get_cusp_without_intp(hue=hue)
        rgb = cs.jzazbz_to_rgb(
            jzazbz=jzczhz_to_jzazbz(cusp), color_space_name=cs.BT709,
            luminance=1000)
        print(f"hue={hue:.2f}, cusp={cusp}, rgb={rgb}")


def create_luts_all():
    chroma_sample = 512
    hue_sample = 4096
    lightness_sample = 1024
    # color_space_name_list = [cs.BT709, cs.BT2020, cs.P3_D65]
    # luminance_list = [100, 300, 600, 1000, 2000, 4000, 10000]
    color_space_name_list = [cs.BT709]
    luminance_list = [1000]

    met = MeasureExecTime()
    met.start()
    for color_space_name in color_space_name_list:
        create_lab_gamut_boundary_method_c(
            hue_sample=hue_sample, lightness_sample=lightness_sample,
            chroma_sample=chroma_sample,
            color_space_name=color_space_name)
    met.end()

    # for color_space_name in color_space_name_list:
    #     for luminance in luminance_list:
    #         create_jzazbz_gamut_boundary_method_c(
    #             hue_sample=hue_sample, lightness_sample=lightness_sample,
    #             chroma_sample=chroma_sample, color_space_name=color_space_name,
    #             luminance=luminance)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_luts_all()
    plot_plane_festival()
    # debug_ng_cusp()

    # debug plot hue angle 250 to 260
    # bg_lut_name = make_jzazbz_gb_lut_fname_method_c(
    #     color_space_name=cs.BT709, luminance=1000)
    # h_val_list = np.linspace(0, 360, 4096)
    # h_val_list2_idx = (h_val_list > 252.5) & (h_val_list < 257.5)
    # for h_idx, h_val in enumerate(h_val_list[h_val_list2_idx]):
    #     plot_cj_plane_with_interpolation_core(
    #         bg_lut_name=bg_lut_name, h_idx=h_idx, h_val=h_val,
    #         color_space_name=cs.BT709, maximum_luminance=1000)

    # bg_lut_name = make_jzazbz_gb_lut_fname_method_c(
    #     color_space_name=cs.BT709, luminance=1000)
    # h_val_list = np.linspace(0, 360, 4096)
    # h_val_list2_idx = (h_val_list > 0) & (h_val_list < 5)
    # for h_idx, h_val in enumerate(h_val_list[h_val_list2_idx]):
    #     plot_cj_plane_with_interpolation_core(
    #         bg_lut_name=bg_lut_name, h_idx=h_idx+1000, h_val=h_val,
    #         color_space_name=cs.BT709, maximum_luminance=1000)
