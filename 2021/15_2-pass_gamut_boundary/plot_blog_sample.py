# -*- coding: utf-8 -*-
"""

```
"""

# import standard libraries
import os
from pathlib import Path
from colour.models.cie_lab import LCHab_to_Lab

# import third-party libraries
import numpy as np
from colour.utilities import tstack
from multiprocessing import Pool, cpu_count

# import my libraries
import plot_utility as pu
import color_space as cs
from create_gamut_booundary_lut import TyLchLut, calc_chroma_boundary_lut
from color_space_plot import create_valid_cielab_ab_plane_image_gm24
from jzazbz import jzczhz_to_jzazbz, jzazbz_to_large_xyz

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

src_dir = "/work/overuse/2021/14_optical_illusion/motion_src/"
i_asset_dir = "/work/overuse/2021/14_optical_illusion/intermediate/"


def get_cielab_bg_lut_name(color_space_name=cs.BT709):
    return f"./lut/lut_sample_1024_1024_32768_{color_space_name}.npy"


def get_jzazbz_bg_lut_name(color_space_name=cs.BT709, luminance=100):
    fname = f"./lut/JzChz_gb-lut_type3_{color_space_name}_{luminance}nits"
    fname += "_jj-1024_hh-4096.npy"
    return fname


def hue_deg_to_boundary_ab(l_val, hue_deg, color_space_name=cs.BT709):
    ty_lch_lut = TyLchLut(
        lut=np.load(get_cielab_bg_lut_name(color_space_name=color_space_name)))

    lh_array = np.array([[l_val, hue_deg]])
    chroma = ty_lch_lut.interpolate(lh_array=lh_array)[0, 1]
    hue_rad = np.deg2rad(hue_deg)
    print(f"rad_now = {hue_rad}")
    ab = (np.cos(hue_rad) * chroma, np.sin(hue_rad) * chroma)
    print(f"end_now = {ab}")

    return ab


def plot_gb_annotate_line_from_hue_deg(
        ax1, l_val, hue_deg, color_space_name=cs.BT709,
        arrowstyle='-|>', linestyle='-', scale_rate=1.0, width=1):
    st_pos = (0, 0)
    ed_pos = hue_deg_to_boundary_ab(
        l_val=l_val, hue_deg=hue_deg, color_space_name=color_space_name)
    # plot_annotate_line(
    #     ax1=ax1, st_pos=st_pos, ed_pos=ed_pos, color=(0.1, 0.1, 0.1),
    #     is_arrow=is_arrow, linestyle=linestyle, scale_rate=scale_rate)
    plot_annotate_line(
        ax1=ax1, st_pos=st_pos, ed_pos=ed_pos, color=(0.1, 0.1, 0.1),
        arrowstyle=arrowstyle, linestyle=linestyle, is_curve=False,
        is_negative=False, text="", scale_rate=scale_rate, width=width)


def conv_deg_to_n180_p180(deg):
    deg_new = deg % 360
    if deg_new > 180:
        deg_new = deg_new - 360

    return deg_new


def plot_arc_for_hue_deg(ax1, hue_deg=60, radius=10):
    hue_deg_new = conv_deg_to_n180_p180(hue_deg)
    is_negative = True if hue_deg_new < 0 else False
    hue_rad = np.deg2rad(hue_deg_new)
    st_pos = (radius, 0)
    ed_pos = (radius * np.cos(hue_rad), radius * np.sin(hue_rad))

    plot_annotate_line(
        ax1, st_pos=st_pos, ed_pos=ed_pos, color=(0.1, 0.1, 0.1),
        is_arrow=True, is_curve=True, is_negative=is_negative)


def plot_annnotate_text_for_hue_deg(ax1, hue_deg=60, radius=10, text=""):
    hue_deg_new = conv_deg_to_n180_p180(hue_deg)
    hue_rad = np.deg2rad(hue_deg_new / 2)
    st_pos = (radius * np.cos(hue_rad), radius * np.sin(hue_rad))

    plot_annotate_line(
        ax1=ax1, st_pos=st_pos, color=(0.1, 0.1, 0.1),
        is_line=False, text=text)


def plot_annotate_line(
        ax1, st_pos=(0, 0), ed_pos=(0, 0), color=(0.1, 0.1, 0.1),
        arrowstyle='-|>', linestyle='-', is_curve=False, is_negative=False,
        text="", scale_rate=1.0, width=1):
    rad = 0.6
    # if is_line:
    #     if is_arrow:
    #         headwidth = 12 * scale_rate
    #         headlength = 16 * scale_rate
    #     else:
    #         headwidth = width
    #         headlength = 0.00001
    # else:
    #     headwidth = 0
    #     headlength = 0

    if is_curve:
        if is_negative:
            connectionstyle = f"arc3,rad=-{rad}"
        else:
            connectionstyle = f"arc3,rad={rad}"

    else:
        connectionstyle = None

    arrowprops = dict(
        facecolor='#000000',
        # headwidth=headwidth, headlength=headlength,
        arrowstyle=arrowstyle, linestyle=linestyle,
        alpha=1.0, connectionstyle=connectionstyle,
    )
    arrowprops['facecolor'] = np.array(color)
    ax1.annotate(
        text=text, xy=ed_pos, xytext=st_pos, xycoords='data',
        textcoords='data', ha='center', va='center',
        arrowprops=arrowprops)
    # ax1.annotate(
    #     "", xy=ed_pos, xytext=st_pos, xycoords='data',
    #     textcoords='data', ha='left', va='bottom',
    #     arrowstyle=None)


def plot_ab_plane(l_val=70, hue_deg=0):
    # l_val = 70
    ab_max = 80
    ab_sample = 512
    hue_num = 361
    hh = np.linspace(0, 360, hue_num)
    ll = np.ones_like(hh) * l_val
    lh_array = tstack([ll, hh])
    print(lh_array.shape)

    ty_lch_lut = TyLchLut(
        lut=np.load("./lut/lut_sample_1024_1024_32768_ITU-R BT.709.npy"))
    lch = ty_lch_lut.interpolate(lh_array=lh_array)
    aa = lch[..., 1] * np.cos(np.deg2rad(lch[..., 2]))
    bb = lch[..., 1] * np.sin(np.deg2rad(lch[..., 2]))

    rgb_img = create_valid_cielab_ab_plane_image_gm24(
        l_val=l_val, ab_max=ab_max, ab_sample=ab_sample,
        color_space_name=cs.BT709,
        bg_rgb_luminance=np.array([18, 18, 18]))

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Title",
        graph_title_size=None,
        xlabel="a", ylabel="b",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        rgb_img, extent=(-ab_max, ab_max, -ab_max, ab_max), aspect='auto')
    ax1.plot(aa, bb, color='k', label=f"L={l_val}")

    # hue_deg = 180
    hue_deg_annnotate_radius = 10
    hue_deg_text_radius = hue_deg_annnotate_radius * 1.4
    plot_gb_annotate_line_from_hue_deg(
        ax1=ax1, l_val=l_val, hue_deg=hue_deg, color_space_name=cs.BT709,
        is_arrow=True)
    plot_gb_annotate_line_from_hue_deg(
        ax1=ax1, l_val=l_val, hue_deg=0, color_space_name=cs.BT709,
        is_arrow=False)
    plot_arc_for_hue_deg(
        ax1=ax1, hue_deg=hue_deg, radius=hue_deg_annnotate_radius)
    plot_annnotate_text_for_hue_deg(
        ax1=ax1, hue_deg=hue_deg, radius=hue_deg_text_radius, text="θ")

    pu.show_and_save(
        fig=fig, legend_loc='upper right', show=False,
        save_fname=f"./img/ab_sample_{hue_deg}.png")


def plot_ab_plane_with_rough_lut(grid=16, l_val=70):
    hue_sample = grid
    ll_num = grid
    lut_name = f"./lut/2dlut_{grid}x{grid}.npy"
    chroma_sample = 32768
    cs_name = cs.BT709
    lut = calc_chroma_boundary_lut(
        lightness_sample=ll_num, chroma_sample=chroma_sample,
        hue_sample=hue_sample, cs_name=cs_name)
    np.save(lut_name, lut)
    ops_bak = np.get_printoptions()
    np.set_printoptions(precision=1)
    print(lut)
    np.set_printoptions(**ops_bak)

    lut = np.load(lut_name)
    np.set_printoptions(precision=2)
    lightness_idx = int(round(l_val / (100 / (grid - 1))))
    print(f"lightness_idx = {lightness_idx}")
    l_val_lut = lut[lightness_idx][..., 0]
    hue_deg = lut[lightness_idx][..., 2]
    hue_num = len(hue_deg)
    chroma = lut[lightness_idx][..., 1]
    hue_rad = np.deg2rad(hue_deg)
    a = chroma * np.cos(hue_rad)
    b = chroma * np.sin(hue_rad)

    ab_max = 80
    ab_sample = 4096

    rgb_img = create_valid_cielab_ab_plane_image_gm24(
        l_val=l_val, ab_max=ab_max, ab_sample=ab_sample,
        color_space_name=cs_name,
        bg_rgb_luminance=np.array([18, 18, 18]))

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=f"a-b plane (L*=70, Grid={grid}x{grid}, Gamut={cs_name})",
        graph_title_size=None,
        xlabel="a", ylabel="b",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        rgb_img, extent=(-ab_max, ab_max, -ab_max, ab_max), aspect='auto')
    ax1.plot(a, b, '-o', color='k', label=f"L*={l_val_lut[0]:.2f}", ms=10)

    for h_idx in range(hue_num):
        plot_gb_annotate_line_from_hue_deg(
            ax1=ax1, l_val=l_val_lut[h_idx], hue_deg=hue_deg[h_idx],
            color_space_name=cs.BT709,
            arrowstyle='-', linestyle='--', scale_rate=1.0, width=1)

    pu.show_and_save(
        fig=fig, legend_loc='upper right', show=False,
        save_fname=f"./img/ab_sample_l_70_{grid}x{grid}.png")


def plot_simple_ab_plane(l_val=70, cs_name=cs.BT709):
    ab_max = 80
    ab_sample = 4096

    ty_lch_lut = TyLchLut(
        lut=np.load(get_cielab_bg_lut_name(color_space_name=cs_name)))
    hh_list = np.linspace(0, 360, 1024)
    ll_list = np.ones_like(hh_list) * l_val
    lh_array = tstack([ll_list, hh_list])
    chroma = ty_lch_lut.interpolate(lh_array=lh_array)[..., 1]
    hue_rad = np.deg2rad(hh_list)
    a = chroma * np.cos(hue_rad)
    b = chroma * np.sin(hue_rad)

    rgb_img = create_valid_cielab_ab_plane_image_gm24(
        l_val=l_val, ab_max=ab_max, ab_sample=ab_sample,
        color_space_name=cs_name,
        bg_rgb_luminance=np.array([18, 18, 18]))

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=f"a-b plane (L*=70, Gamut={cs_name})",
        graph_title_size=None,
        xlabel="a", ylabel="b",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        rgb_img, extent=(-ab_max, ab_max, -ab_max, ab_max), aspect='auto')
    ax1.plot(a, b, '-', color='k', label="Gamut Boundary")

    pu.show_and_save(
        fig=fig, legend_loc='lower right', show=False,
        save_fname="./img/ab_sample_l_70.png")


def _conv_Lab_to_abL(Lab):
    abL = Lab.copy()
    abL[..., 0] = Lab[..., 1]
    abL[..., 1] = Lab[..., 2]
    abL[..., 2] = Lab[..., 0]

    return abL


def plot_gamut_boundary_cielab():
    idx_num = 360

    total_process_num = idx_num
    block_process_num = cpu_count() // 2
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            main_index = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={main_index}")  # User
            if main_index >= total_process_num:                         # User
                break
            d = dict(idx=main_index)
            args.append(d)
        #     break
        # break
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_plot_gamut_boundary_cielab_core, args)


def thread_wrapper_plot_gamut_boundary_cielab_core(args):
    plot_gamut_boundary_cielab_core(**args)


def plot_gamut_boundary_cielab_core(idx):
    color_space_name = cs.BT709
    lch_lut = np.load(
        get_cielab_bg_lut_name(color_space_name=color_space_name))
    lab_lut = LCHab_to_Lab(lch_lut).reshape(1024*1024, 3)
    lab = lab_lut
    abl = _conv_Lab_to_abL(Lab=lab)
    # bgd = GamutBoundaryData(Lab=lab_lut)
    # abl = bgd.get_outline_mesh_data_as_abL(
    #     ab_plane_div_num=50, rad_rate=5, l_step=5)
    # lab = bgd._conv_abL_to_Lab(abL=abl)
    rgb = cs.lab_to_rgb(lab=lab, color_space_name=cs.BT709)
    rgb_gm24 = np.clip(rgb, 0.0, 1.0) ** (1/2.4)

    fig, ax = pu.plot_3d_init(
        figsize=(10, 10),
        title="CIELAB BT.709 Gamut Boundary",
        title_font_size=18,
        face_color=(0.1, 0.1, 0.1),
        plane_color=(0.2, 0.2, 0.2, 1.0),
        text_color=(0.5, 0.5, 0.5),
        grid_color=(0.3, 0.3, 0.3),
        x_label="a",
        y_label="b",
        z_label="L*",
        xlim=[-120, 120],
        ylim=[-120, 120],
        zlim=[-5, 105])
    pu.plot_xyY_with_scatter3D(ax, abl, color=rgb_gm24)
    ax.view_init(elev=20, azim=-120+idx)
    fname = "/work/overuse/2021/15_2_pass_gamut_boundary/img_seq_cielab/"
    fname += f"cielab_3d_bg709_{idx:04d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, save_fname=fname, show=False)


def thread_wrapper_plot_gamut_boundary_jzazbz_core(args):
    plot_gamut_boundary_jzazbz_core(**args)


def plot_gamut_boundary_jzazbz():
    idx_num = 360

    total_process_num = idx_num
    block_process_num = cpu_count() // 2
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            main_index = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={main_index}")  # User
            if main_index >= total_process_num:                         # User
                break
            d = dict(idx=main_index)
            args.append(d)
        #     break
        # break
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_plot_gamut_boundary_jzazbz_core, args)


def plot_gamut_boundary_jzazbz_core(idx=0):
    color_space_name = cs.BT709
    luminance = 100
    jzczhz_lut = np.load(
        get_jzazbz_bg_lut_name(
            color_space_name=color_space_name, luminance=luminance))
    jzazbz_lut = jzczhz_to_jzazbz(jzczhz_lut).reshape(1024*4096, 3)
    jzazbz = jzazbz_lut
    azbzczjz = _conv_Lab_to_abL(Lab=jzazbz)
    rgb = cs.jzazbz_to_rgb(
        jzazbz=jzazbz, color_space_name=color_space_name,
        luminance=luminance)
    rgb_gm24 = np.clip(rgb, 0.0, 1.0) ** (1/2.4)

    fig, ax = pu.plot_3d_init(
        figsize=(10, 10),
        title="Jzazbz BT.709 Gamut Boundary",
        title_font_size=18,
        face_color=(0.1, 0.1, 0.1),
        plane_color=(0.2, 0.2, 0.2, 1.0),
        text_color=(0.5, 0.5, 0.5),
        grid_color=(0.3, 0.3, 0.3),
        x_label="a",
        y_label="b",
        z_label="L*",
        xlim=[-0.21, 0.21],
        ylim=[-0.21, 0.21],
        zlim=[-0.02, 0.2])
    pu.plot_xyY_with_scatter3D(ax, azbzczjz, color=rgb_gm24)
    ax.view_init(elev=20, azim=-120+idx)
    fname = "/work/overuse/2021/15_2_pass_gamut_boundary/img_seq_jzazbz/"
    fname += f"jzazbz_3d_bg709_{idx:04d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, save_fname=fname, show=False)


def plot_color_volume_cielab_rough(grid=8, idx=0):
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # hue_deg_list = np.linspace(0, 360, 16, endpoint=False)
    # for deg in hue_deg_list:
    #     plot_ab_plane(hue_deg=deg)
    # plot_simple_ab_plane()
    # plot_ab_plane_with_rough_lut(grid=8)
    # plot_ab_plane_with_rough_lut(grid=16)
    # plot_ab_plane_with_rough_lut(grid=32)
    # plot_ab_plane_with_rough_lut(grid=64)
    # plot_gamut_boundary_cielab()
    # plot_gamut_boundary_jzazbz_core(idx=0)
    plot_gamut_boundary_jzazbz()
