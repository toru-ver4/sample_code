# -*- coding: utf-8 -*-
"""

```
"""

# import standard libraries
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

# import third-party libraries
import numpy as np
from colour import Lab_to_XYZ, XYZ_to_RGB
from colour.models import RGB_COLOURSPACES
from colour.utilities import tstack
from cielab import is_inner_gamut

# import my libraries
import plot_utility as pu
import color_space as cs
import transfer_functions as tf
from create_gamut_booundary_lut import JZAZBZ_CHROMA_MAX, is_out_of_gamut_rgb,\
    create_jzazbz_gamut_boundary_lut_type2,\
    make_jzazbz_gb_lut_fname_methodb_b, make_jzazbz_gb_lut_fname_method_c,\
    TyLchLut
from color_space_plot import create_valid_jzazbz_ab_plane_image_st2084,\
    create_valid_jzazbz_cj_plane_image_st2084

from jzazbz import jzazbz_to_large_xyz, jzczhz_to_jzazbz

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


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

    trial_num = 20

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


def debug_plot_azbz_plane_with_interpolation(
        hue_sample=256, lightness_sample=256, j_num_intp=256,
        color_space_name=cs.BT2020, luminance=10000):

    lut_name = make_jzazbz_gb_lut_fname_method_c(
        color_space_name=color_space_name, luminance=luminance,
        lightness_num=lightness_sample, hue_num=hue_sample)
    bg_lut = TyLchLut(np.load(lut_name))
    j_max = bg_lut.ll_max

    j_num = j_num_intp

    total_process_num = j_num
    block_process_num = int(cpu_count() / 3 + 0.999)
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            j_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={j_idx}")  # User
            if j_idx >= total_process_num:                         # User
                break
            # j_idx = j_num - 1
            d = dict(
                bg_lut_name=lut_name, j_idx=j_idx,
                j_val=j_idx/(j_num-1) * j_max,
                color_space_name=color_space_name,
                maximum_luminance=luminance)
            # plot_ab_plane_with_interpolation_core(**d)
            args.append(d)
        #     break
        # break
        with Pool(block_process_num) as pool:
            pool.map(thread_wrapper_plot_ab_plane_with_interpolation, args)


def thread_wrapper_plot_ab_plane_with_interpolation(args):
    plot_ab_plane_with_interpolation_core(**args)


def plot_ab_plane_with_interpolation_core(
        bg_lut_name, j_idx, j_val, color_space_name, maximum_luminance):
    if maximum_luminance <= 101:
        ab_max = 0.25
    elif maximum_luminance <= 1001:
        ab_max = 0.30
    else:
        ab_max = 0.50
    ab_sample = 1024
    hue_sample = 1024
    bg_lut = TyLchLut(np.load(bg_lut_name))
    rgb_st2084 = create_valid_jzazbz_ab_plane_image_st2084(
        j_val=j_val, ab_max=ab_max, ab_sample=ab_sample,
        color_space_name=color_space_name,
        bg_rgb_luminance=np.array([100, 100, 100]),
        maximum_luminance=maximum_luminance)

    hh_base = np.linspace(0, 360, hue_sample)
    jj_base = np.ones_like(hh_base) * j_val
    jh_array = tstack([jj_base, hh_base])

    jzczhz = bg_lut.interpolate(lh_array=jh_array)
    chroma = jzczhz[..., 1]
    hue = np.deg2rad(jzczhz[..., 2])
    aa = chroma * np.cos(hue)
    bb = chroma * np.sin(hue)

    jzazbz_luminance = jzczhz_to_jzazbz(jzczhz[0])
    xyz_lumi = jzazbz_to_large_xyz(jzazbz_luminance)
    rgb_lumi = XYZ_to_RGB(
        xyz_lumi/100, cs.D65, cs.D65,
        RGB_COLOURSPACES[cs.BT709].matrix_XYZ_to_RGB)
    # print(f"jzczhz={jzczhz[0]}")
    # print(f"xyz={xyz_lumi}")
    # print(f"rgb={rgb_lumi}")
    # lut_b = np.load(
    #     make_jzazbz_gb_lut_fname_methodb_b(
    #         color_space_name=color_space_name,
    #         luminance=maximum_luminance,
    #         lightness_num=256, hue_num=256))
    # print(f"lut={lut_b[0]}")
    # print(f"lut_c={bg_lut.lut[-1]}")

    large_xyz = jzazbz_to_large_xyz(jzazbz_luminance)
    luminance = large_xyz[1]
    graph_title = f"azbz plane,  {color_space_name},  "
    graph_title += f"Jz={j_val:.2f},  Luminance={luminance:.2f} nits"
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=graph_title,
        graph_title_size=None,
        xlabel="az", ylabel="bz",
        axis_label_size=None,
        legend_size=17,
        xlim=[-ab_max, ab_max],
        ylim=[-ab_max, ab_max],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        rgb_st2084, extent=(-ab_max, ab_max, -ab_max, ab_max), aspect='auto')
    ax1.plot(aa, bb, color='k')
    fname = "/work/overuse/2021/15_2_pass_gamut_boundary/img_seq_azbz/"
    fname += f"azbz_w_lut_{color_space_name}_"
    fname += f"{maximum_luminance}nits_{j_idx:04d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)


def thread_wrapper_plot_cj_plane_with_interpolation(args):
    plot_cj_plane_with_interpolation_core(**args)


def plot_cj_plane_with_interpolation_core(
        bg_lut_name, h_idx, h_val, color_space_name, maximum_luminance):
    bg_lut = TyLchLut(lut=np.load(bg_lut_name))
    sample_num = 1024
    jj_sample = 1024
    if maximum_luminance <= 101:
        cc_max = 0.25
        jj_max = 0.2
    elif maximum_luminance <= 1001:
        cc_max = 0.3
        jj_max = 0.5
    else:
        cc_max = 0.5
        jj_max = 1.0
    print(f"h_val={h_val} started")

    rgb_st2084 = create_valid_jzazbz_cj_plane_image_st2084(
        h_val=h_val, c_max=cc_max, l_max=jj_max,
        c_sample=sample_num, j_sample=sample_num,
        color_space_name=color_space_name,
        bg_rgb_luminance=np.array([100, 100, 100]),
        maximum_luminance=maximum_luminance)
    graph_title = f"CzJz plane,  {color_space_name},  hue={h_val:.2f}°,  "
    graph_title += f"target={maximum_luminance} nits"

    jj_base = np.linspace(0, bg_lut.ll_max, jj_sample)
    hh_base = np.ones_like(jj_base) * h_val
    jh_array = tstack([jj_base, hh_base])
    # jzczhz = get_gamut_boundary_lch_from_lut(
    #     lut=bg_lut, lh_array=jh_array, lightness_max=1.0)
    jzczhz = bg_lut.interpolate(lh_array=jh_array)

    chroma = jzczhz[..., 1]
    lightness = jzczhz[..., 0]

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=graph_title,
        graph_title_size=None,
        xlabel="Cz", ylabel="Jz",
        axis_label_size=None,
        legend_size=17,
        xlim=[0, cc_max],
        ylim=[0, jj_max],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        rgb_st2084, extent=(0, cc_max, 0, jj_max), aspect='auto')
    ax1.plot(chroma, lightness, color='k')
    fname = "/work/overuse/2021/15_2_pass_gamut_boundary/img_seq_czjz/"
    fname += f"CzJz_w_lut_{color_space_name}_{maximum_luminance}-nits_"
    fname += f"{h_idx:04d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)


def plot_cj_plane_with_interpolation(
        hue_sample=256, lightness_sample=256, h_num_intp=256,
        color_space_name=cs.BT2020, luminance=10000):

    lut_name = make_jzazbz_gb_lut_fname_method_c(
        color_space_name=color_space_name, luminance=luminance,
        lightness_num=lightness_sample, hue_num=hue_sample)

    total_process_num = h_num_intp
    block_process_num = int(cpu_count() / 3 + 0.9)
    print(f"block_process_num {block_process_num}")
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            h_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={h_idx}")  # User
            if h_idx >= total_process_num:                         # User
                break
            d = dict(
                bg_lut_name=lut_name, h_idx=h_idx,
                h_val=h_idx/(h_num_intp-1)*360,
                color_space_name=color_space_name,
                maximum_luminance=luminance)
            # plot_cj_plane_with_interpolation_core(**d)
            args.append(d)
        #     break
        # break
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_plot_cj_plane_with_interpolation, args)


def debug_plot_jzazbz():
    chroma_sample = 4
    hue_sample = 256
    lightness_sample = 128
    luminance = 100
    h_num_intp = 100
    j_num_intp = 100
    color_space_name = cs.BT709
    create_jzazbz_gamut_boundary_method_c(
        hue_sample=hue_sample, lightness_sample=lightness_sample,
        chroma_sample=chroma_sample,
        color_space_name=color_space_name, luminance=luminance)
    debug_plot_azbz_plane_with_interpolation(
        hue_sample=hue_sample, lightness_sample=lightness_sample,
        j_num_intp=j_num_intp,
        color_space_name=color_space_name, luminance=luminance)
    plot_cj_plane_with_interpolation(
        hue_sample=hue_sample, lightness_sample=lightness_sample,
        h_num_intp=h_num_intp,
        color_space_name=color_space_name, luminance=luminance)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_plot_jzazbz()
