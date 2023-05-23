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

# import my libraries
import plot_utility as pu
import color_space as cs
from create_gamut_booundary_lut import CIELAB_CHROMA_MAX, TyLchLut,\
    create_cielab_gamut_boundary_lut_method_b,\
    calc_chroma_boundary_specific_ligheness_cielab_method_c
from common import MeasureExecTime
from color_space_plot import create_valid_oklab_ab_plane_image_gm24,\
    create_valid_oklab_cl_plane_image_gm24


OKLAB_CHROMA_MAX = 1.0

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def make_oklab_gb_lut_fname_method_b(
        color_space_name, lightness_num, hue_num):
    fname = f"./lut/oklab_gb-lut_method_b_{color_space_name}_"
    fname += f"ll-{lightness_num}_hh-{hue_num}.npy"

    return fname


def make_oklab_gb_lut_fname_method_c(
        color_space_name, lightness_num, hue_num):
    fname = f"./lut/oklab_gb-lut_method_c_{color_space_name}_"
    fname += f"ll-{lightness_num}_hh-{hue_num}.npy"

    return fname


def create_lab_gamut_boundary_method_c(
        hue_sample=8, lightness_sample=8, chroma_sample=1024,
        color_space_name=cs.BT709):

    ll_num = lightness_sample
    lut = create_cielab_gamut_boundary_lut_method_b(
        lightness_sample=ll_num, chroma_sample=chroma_sample,
        hue_sample=hue_sample, cs_name=color_space_name, ll_max=1.0,
        lab_to_rgb_func=cs.oklab_to_rgb, chroma_max=OKLAB_CHROMA_MAX)
    np.save(make_oklab_gb_lut_fname_method_b(
        color_space_name=color_space_name, lightness_num=lightness_sample,
        hue_num=hue_sample), lut)
    # print(lut)

    # create 2d lut using method B
    lut_b = np.load(
        make_oklab_gb_lut_fname_method_b(
            color_space_name=color_space_name, lightness_num=lightness_sample,
            hue_num=hue_sample))

    # create 2d lut using method C
    c0 = CIELAB_CHROMA_MAX / (chroma_sample - 1)
    lut_c = np.zeros_like(lut_b)
    for l_idx in range(lightness_sample):
        lch_init = lut_b[l_idx]
        lch_result = calc_chroma_boundary_specific_ligheness_cielab_method_c(
            lch=lch_init, cs_name=color_space_name, c0=c0,
            lab_to_rgb_func=cs.oklab_to_rgb)
        lut_c[l_idx] = lch_result

    fname = make_oklab_gb_lut_fname_method_c(
        color_space_name=color_space_name,
        lightness_num=lightness_sample, hue_num=hue_sample)
    np.save(fname, np.float32(lut_c))


def debug_plot_cielab_ab_plane_with_interpolation(
        hue_sample=256, lightness_sample=256, l_num_intp=256,
        color_space_name=cs.BT2020):

    lut_name = make_oklab_gb_lut_fname_method_c(
        color_space_name=color_space_name,
        lightness_num=lightness_sample, hue_num=hue_sample)
    ll_max = 1.0
    ll_num = l_num_intp

    total_process_num = ll_num
    block_process_num = int(cpu_count() / 2 + 0.999)
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            l_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={l_idx}")  # User
            if l_idx >= total_process_num:                         # User
                break
            # j_idx = j_num - 1
            d = dict(
                bg_lut_name=lut_name, l_idx=l_idx,
                l_val=l_idx/(ll_num-1) * ll_max,
                color_space_name=color_space_name)
            # plot_ab_plane_with_interpolation_core(**d)
            args.append(d)
        #     break
        # break
        with Pool(block_process_num) as pool:
            pool.map(thread_wrapper_plot_ab_plane_with_interpolation, args)


def thread_wrapper_plot_ab_plane_with_interpolation(args):
    plot_ab_plane_with_interpolation_core(**args)


def plot_ab_plane_with_interpolation_core(
        bg_lut_name, l_idx, l_val, color_space_name):
    if color_space_name == cs.BT709:
        ab_max = 0.45
    elif color_space_name == cs.P3_D65:
        ab_max = 0.45
    else:
        ab_max = 0.45
    ab_sample = 2048
    hue_sample = 4000
    bg_lut = TyLchLut(np.load(bg_lut_name))
    rgb_gm24 = create_valid_oklab_ab_plane_image_gm24(
        l_val=l_val, ab_max=ab_max, ab_sample=ab_sample,
        color_space_name=color_space_name,
        bg_rgb_luminance=np.array([0.5, 0.5, 0.5]))

    hh_base = np.linspace(0, 360, hue_sample)
    ll_base = np.ones_like(hh_base) * l_val
    lh_array = tstack([ll_base, hh_base])

    lch = bg_lut.interpolate(lh_array=lh_array)
    chroma = lch[..., 1]
    hue = np.deg2rad(lch[..., 2])
    aa = chroma * np.cos(hue)
    bb = chroma * np.sin(hue)

    graph_title = f"Oklab ab plane,  {color_space_name},  l={l_val:.3f}"
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 12),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=graph_title,
        graph_title_size=None,
        xlabel="a", ylabel="b",
        axis_label_size=None,
        legend_size=17,
        xlim=[-ab_max, ab_max],
        ylim=[-ab_max, ab_max],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=1,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        rgb_gm24, extent=(-ab_max, ab_max, -ab_max, ab_max), aspect='auto')
    ax1.plot(aa, bb, color='k')
    fname = "/work/overuse/2023/05_oog_color_map/oklab_ab_low/"
    fname += f"ab_w_lut_{color_space_name}_{l_idx:04d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)


def thread_wrapper_plot_oklab_cl_plane_with_interpolation(args):
    plot_oklab_cl_plane_with_interpolation_core(**args)


def plot_oklab_cl_plane_with_interpolation_core(
        bg_lut_name, h_idx, h_val, color_space_name):
    bg_lut = TyLchLut(lut=np.load(bg_lut_name))
    sample_num = 1536
    jj_sample = 1536
    ll_max = 1.0
    if color_space_name == cs.BT709:
        cc_max = 0.45
    elif color_space_name == cs.P3_D65:
        cc_max = 0.45
    else:
        cc_max = 0.45
    print(f"h_val={h_val} started")

    rgb_gm24 = create_valid_oklab_cl_plane_image_gm24(
        h_val=h_val, c_max=cc_max, c_sample=sample_num, l_sample=sample_num,
        color_space_name=color_space_name, bg_val=0.5)
    graph_title = f"CL plane,  {color_space_name},  Hue={h_val:.2f}°,  "

    ll_base = np.linspace(0, bg_lut.ll_max, jj_sample)
    hh_base = np.ones_like(ll_base) * h_val
    lh_array = tstack([ll_base, hh_base])
    lch = bg_lut.interpolate(lh_array=lh_array)

    chroma = lch[..., 1]
    lightness = lch[..., 0]

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 12),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=graph_title,
        graph_title_size=None,
        xlabel="C", ylabel="L",
        axis_label_size=None,
        legend_size=17,
        xlim=[0, cc_max],
        ylim=[0, ll_max],
        xtick=None,
        ytick=[x * 0.1 for x in range(11)],
        xtick_size=None, ytick_size=None,
        linewidth=1,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        rgb_gm24, extent=(0, cc_max, 0, ll_max), aspect='auto')
    ax1.plot(chroma, lightness, color='k')
    fname = "/work/overuse/2023/05_oog_color_map/oklab_cl/"
    fname += f"ok_CL_w_lut_{color_space_name}_{h_idx:04d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)


def debug_plot_cielab_cl_plane_with_interpolation(
        hue_sample=256, lightness_sample=256, h_num_intp=256,
        color_space_name=cs.BT2020):

    lut_name = make_oklab_gb_lut_fname_method_c(
        color_space_name=color_space_name,
        lightness_num=lightness_sample, hue_num=hue_sample)

    total_process_num = h_num_intp
    block_process_num = int(cpu_count() / 4 + 0.9)
    print(f"block_process_num {block_process_num}")
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            h_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, h_idx={h_idx}")  # User
            if h_idx >= total_process_num:                         # User
                break
            d = dict(
                bg_lut_name=lut_name, h_idx=h_idx,
                h_val=h_idx/(h_num_intp-1)*360,
                color_space_name=color_space_name)
            # plot_cielab_cj_plane_with_interpolation_core(**d)
            args.append(d)
        #     break
        # break
        with Pool(cpu_count()) as pool:
            pool.map(
                thread_wrapper_plot_oklab_cl_plane_with_interpolation, args)


def plot_oklab(
        hue_sample, lightness_sample, color_space_name,
        l_num_intp, h_num_intp):
    # debug_plot_cielab_ab_plane_with_interpolation(
    #     hue_sample=hue_sample, lightness_sample=lightness_sample,
    #     l_num_intp=l_num_intp, color_space_name=color_space_name)
    debug_plot_cielab_cl_plane_with_interpolation(
        hue_sample=hue_sample, lightness_sample=lightness_sample,
        h_num_intp=h_num_intp, color_space_name=color_space_name)


def create_oklab_luts_all():
    chroma_sample = 256
    hue_sample = 4096
    lightness_sample = 1024
    # color_space_name_list = [cs.BT709, cs.P3_D65, cs.BT2020]
    # color_space_name_list = [cs.BT2020]
    color_space_name_list = [cs.BT709]

    met = MeasureExecTime()
    met.start()
    for color_space_name in color_space_name_list:
        # create_lab_gamut_boundary_method_c(
        #     hue_sample=hue_sample, lightness_sample=lightness_sample,
        #     chroma_sample=chroma_sample,
        #     color_space_name=color_space_name)
        plot_oklab(
            hue_sample=hue_sample, lightness_sample=lightness_sample,
            color_space_name=color_space_name, l_num_intp=1000,
            h_num_intp=1000)
    met.end()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_oklab_luts_all()
