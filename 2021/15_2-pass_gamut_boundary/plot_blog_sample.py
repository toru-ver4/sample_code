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
from create_gamut_booundary_lut import TyLchLut, create_cielab_gamut_boundary_lut_method_b,\
    is_out_of_gamut_rgb
from color_space_plot import create_valid_cielab_ab_plane_image_gm24,\
    create_valid_jzazbz_ab_plane_image_gm24
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
        is_curve=True, is_negative=is_negative)


def plot_annnotate_text_for_hue_deg(
        ax1, hue_deg=60, radius=10, text="",
        rate=1.02, ha='right', va='center'):
    hue_deg_new = conv_deg_to_n180_p180(hue_deg)
    hue_rad = np.deg2rad(hue_deg_new / 2)
    pos = (radius * np.cos(hue_rad), radius * np.sin(hue_rad))

    # plot_annotate_line(
    #     ax1=ax1, st_pos=st_pos, color=(0.1, 0.1, 0.1), text=text)
    plot_annotate_text_only(
        ax1=ax1, pos=pos, text=text, rate=rate, ha=ha, va=va)


def plot_annotate_line(
        ax1, st_pos=(0, 0), ed_pos=(0, 0), color=(0.1, 0.1, 0.1),
        arrowstyle='-|>', linestyle='-', is_curve=False, is_negative=False,
        text=None, alpha=1.0):
    rad = 0.6
    if is_curve:
        if is_negative:
            connectionstyle = f"arc3,rad=-{rad}"
        else:
            connectionstyle = f"arc3,rad={rad}"
    else:
        connectionstyle = None

    arrowprops = dict(
        facecolor=color,
        edgecolor=color,
        arrowstyle=arrowstyle, linestyle=linestyle,
        alpha=alpha, connectionstyle=connectionstyle,
    )
    ax1.annotate(
        text=text, xy=ed_pos, xytext=st_pos, xycoords='data',
        textcoords='data', ha='center', va='center',
        arrowprops=arrowprops)


def plot_annotate_text_only(
        ax1=None, pos=(0, 0), text="hoge", rate=1.01, ha='left', va='bottom'):
    x_min, x_max = ax1.get_xlim()
    x_len = x_max - x_min
    y_min, y_max = ax1.get_ylim()
    y_len = y_max - y_min

    text_x = pos[0]
    text_y = pos[1]
    if ha == 'left':
        text_x += x_len * (rate - 1)
    elif ha == 'right':
        text_x -= x_len * (rate - 1)
    else:
        None

    if va == 'bottom':
        text_y += y_len * (rate - 1)
    elif va == 'top':
        text_y -= y_len * (rate - 1)

    ax1.annotate(text, xy=pos, xytext=(text_x, text_y), ha=ha, va=va)


def hue_chroma_to_ab(hue_deg=180, chroma=60):
    """
    Examples
    --------
    >>> hue_chroma_to_ab(hue_deg=180, chroma=60)
    -60.0, 7.3478807948841199e-15
    >>> hue_deg = np.linspace(0, 360, 8)
    >>> chroma = np.linspace(50, 100, 8)
    >>> hue_chroma_to_ab(hue_deg=hue_deg, chroma=chroma)
    [50., 35.62798868, -14.30491718, -64.35491914, -70.79041105,
     -19.07322291,   57.8954816 ,  100.],
    [0.00000000e+00, 4.46760847e+01, 6.26739372e+01, 3.09916957e+01,
     -3.40908652e+01, -8.35652496e+01, -7.25986377e+01,  -2.44929360e-14]
    """
    rad = np.deg2rad(hue_deg)

    a = chroma * np.cos(rad)
    b = chroma * np.sin(rad)

    return a, b


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
    lut = create_cielab_gamut_boundary_lut_method_b(
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


def plot_color_volume_cielab_rough_core(grid=8, idx=0):
    lut_name = f"./lut/2dlut_{grid}x{grid}.npy"
    color_space_name = cs.BT709
    lch_lut = np.load(lut_name)
    lab_lut = LCHab_to_Lab(lch_lut)
    lab = lab_lut
    abl = _conv_Lab_to_abL(Lab=lab)
    rgb = cs.lab_to_rgb(lab=lab, color_space_name=color_space_name)
    rgb_gm24 = np.clip(rgb, 0.0, 1.0) ** (1/2.4)

    fig, ax = pu.plot_3d_init(
        figsize=(10, 10),
        title=f"CIELAB BT.709 Gamut Boundary ({grid}x{grid} 2DLUT)",
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

    for l_idx in range(grid):
        abl_l_idx = abl[l_idx]
        rgb_l_idx = rgb_gm24[l_idx]
        pu.plot_xyY_with_scatter3D(
            ax, abl_l_idx, color=rgb_l_idx, ms=50, edgecolors='k')

        for h_idx in range(len(abl_l_idx)):
            ax.arrow3D(
                0, 0, abl_l_idx[h_idx, 2],
                abl_l_idx[h_idx, 0],
                abl_l_idx[h_idx, 1],
                abl_l_idx[h_idx, 2],
                mutation_scale=16, arrowstyle="-", linestyle='--')
    ax.view_init(elev=20, azim=-120+idx)
    fname = "/work/overuse/2021/15_2_pass_gamut_boundary/3d_rough_lut/"
    fname += f"cielab_3d_rough_bg709_grid-{grid}_{idx:04d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, save_fname=fname, show=False)


def thread_wrapper_plot_color_volume_cielab_rough_core(args):
    plot_color_volume_cielab_rough_core(**args)


def plot_color_volume_cielab_rough(grid=8):
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
            d = dict(grid=grid, idx=main_index)
            args.append(d)
            # plot_color_volume_cielab_rough_core(**d)
        #     break
        # break
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_plot_color_volume_cielab_rough_core, args)


def calc_chroma_candidate_list(
        r_val_init=160, lightness=100, hue_sample=8, cs_name=cs.BT709):
    """
    """
    # lch --> rgb
    trial_num = 30

    hue = np.linspace(0, 2*np.pi, hue_sample)
    r_val = r_val_init * np.ones_like(hue)
    r_temp = np.zeros((trial_num + 1, hue_sample))
    r_temp[0] = r_val_init
    ll = lightness * np.ones_like(hue)

    for t_idx in range(trial_num):
        aa = r_val * np.cos(hue)
        bb = r_val * np.sin(hue)
        lab = tstack((ll, aa, bb))
        rgb = cs.lab_to_rgb(lab=lab, color_space_name=cs_name)

        ng_idx = is_out_of_gamut_rgb(rgb=rgb)
        ok_idx = np.logical_not(ng_idx)
        add_sub = r_val_init / (2 ** (t_idx + 1))
        r_val[ok_idx] = r_val[ok_idx] + add_sub
        r_val[~ok_idx] = r_val[~ok_idx] - add_sub
        r_temp[t_idx + 1] = r_val

    # jzczhz = tstack([jj, r_val, np.rad2deg(hue)])

    return r_temp


def calc_chroma_candidate_list_method_c(
        r_val_init=1.0, c0=0.5, lightness=100, hue_sample=8, cs_name=cs.BT709):
    """
    """
    # lch --> rgb
    trial_num = 20

    hue = np.linspace(0, 2*np.pi, hue_sample)
    r_val = r_val_init * np.ones_like(hue)
    r_temp = np.zeros((trial_num + 1, hue_sample))
    r_temp[0] = r_val_init
    ll = lightness * np.ones_like(hue)

    for t_idx in range(trial_num):
        aa = r_val * np.cos(hue)
        bb = r_val * np.sin(hue)
        lab = tstack((ll, aa, bb))
        rgb = cs.lab_to_rgb(lab=lab, color_space_name=cs_name)

        ng_idx = is_out_of_gamut_rgb(rgb=rgb)
        ok_idx = np.logical_not(ng_idx)
        add_sub = c0 / (2 ** (t_idx + 0))  # NOT "t_idx + 1"
        r_val[ok_idx] = r_val[ok_idx] + add_sub
        r_val[~ok_idx] = r_val[~ok_idx] - add_sub
        r_temp[t_idx + 1] = r_val

    # jzczhz = tstack([jj, r_val, np.rad2deg(hue)])

    return r_temp


def plot_length_line_marker(
        ax1, aa_list, bb_list, idx_list=[-1, 2, 4], offset_x=0, offset_y=0):
    plot_length_line_marker_core(
        ax1, aa_list, bb_list, idx_list[0], idx_list[1], offset_x, offset_y)
    plot_length_line_marker_core(
        ax1, aa_list, bb_list, idx_list[1], idx_list[2], offset_x, offset_y)

    for idx in idx_list:
        plot_vertical_aux_line(
            ax1, aa_list, bb_list, idx, offset_x, offset_y)


def plot_length_line_marker_core(
        ax1, aa_list, bb_list, list_idx1, list_idx2, offset_x, offset_y):
    if list_idx1 < 0:
        base_x = 0
        base_y = 0
    else:
        base_x = aa_list[list_idx1]
        base_y = bb_list[list_idx1]
    plot_annotate_line(
        ax1=ax1,
        st_pos=(base_x+offset_x, base_y+offset_y),
        ed_pos=(aa_list[list_idx2]+offset_x, bb_list[list_idx2]+offset_y),
        color=(0.1, 0.1, 0.1), arrowstyle='<->')


def plot_vertical_aux_line(
        ax1, aa_list, bb_list, list_idx, offset_x, offset_y):
    if list_idx < 0:
        base_x = 0
        base_y = 0
    else:
        base_x = aa_list[list_idx]
        base_y = bb_list[list_idx]

    plot_annotate_line(
        ax1=ax1,
        st_pos=(base_x, base_y),
        ed_pos=(base_x+offset_x, base_y+offset_y),
        color=(0.1, 0.1, 0.1), arrowstyle='-', linestyle='--')


def plot_method_a(grid=8, hue_idx=1):
    hue_sample = grid
    chroma_init = 100
    h_val = np.linspace(0, 360, hue_sample)[hue_idx]
    x_max = int(chroma_init * np.cos(np.deg2rad(h_val)) * 1.2)
    y_max = int(chroma_init * np.sin(np.deg2rad(h_val)) * 1.2)
    xy_max = max([x_max, y_max])
    l_val_dummy = 70
    lut_name = f"./lut/2dlut_{grid}x{grid}.npy"
    cs_name = cs.BT709
    ops_bak = np.get_printoptions()
    np.set_printoptions(precision=1)
    np.set_printoptions(**ops_bak)

    lut = np.load(lut_name)
    np.set_printoptions(precision=2)
    lightness_idx = int(round(l_val_dummy / (100 / (grid - 1))))
    print(f"lightness_idx = {lightness_idx}")
    l_val_lut = lut[lightness_idx][..., 0]
    l_val = l_val_lut[0]

    ab_max = 80
    ab_sample = 1024

    bl = 0.96 ** 2.4  # background luminance
    rgb_img = create_valid_cielab_ab_plane_image_gm24(
        l_val=l_val, ab_max=ab_max, ab_sample=ab_sample,
        color_space_name=cs_name,
        bg_rgb_luminance=np.array([bl, bl, bl]))

    cc_list = calc_chroma_candidate_list(
        r_val_init=chroma_init, lightness=l_val, hue_sample=grid,
        cs_name=cs.BT709)[..., hue_idx]
    # aa_list = cc_list * np.cos(np.deg2rad(h_val))
    # bb_list = cc_list * np.sin(np.deg2rad(h_val))
    aa_list, bb_list = hue_chroma_to_ab(hue_deg=h_val, chroma=cc_list)

    title = f"a-b plane (L*={l_val:.2f}, Grid={grid}x{grid}, Gamut={cs_name})"

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=title,
        graph_title_size=None,
        xlabel="a", ylabel="b",
        axis_label_size=None,
        legend_size=17,
        xlim=[-80, xy_max],
        ylim=[-80, xy_max],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        rgb_img, extent=(-ab_max, ab_max, -ab_max, ab_max), aspect='auto')

    ms = 8
    text_rate = 1.02
    ax1.plot(aa_list[:4], bb_list[:4], 'ko', ms=ms)

    plot_annotate_line(
        ax1=ax1, st_pos=(0, 0), ed_pos=(xy_max, 0), color=(0.1, 0.1, 0.1),
        arrowstyle='-')

    plot_annotate_line(
        ax1=ax1, st_pos=(0, 0), ed_pos=(aa_list[0], bb_list[0]),
        color=(0.1, 0.1, 0.1), arrowstyle='-')

    plot_annotate_text_only(
        ax1=ax1, pos=(aa_list[0], bb_list[0]), rate=text_rate,
        text=r"$P_{j,k}$", ha='left', va='center')
    plot_annotate_text_only(
        ax1=ax1, pos=(aa_list[1], bb_list[1]), rate=text_rate,
        text=r"$P_{j,k+1}$", ha='left', va='center')
    plot_annotate_text_only(
        ax1=ax1, pos=(aa_list[2], bb_list[2]), rate=text_rate,
        text=r"$P_{j,k+2}$", ha='left', va='center')
    plot_annotate_text_only(
        ax1=ax1, pos=(aa_list[3], bb_list[3]), rate=text_rate,
        text=r"$P_{j,k+3}$", ha='left', va='center')

    radius = 13
    plot_arc_for_hue_deg(ax1=ax1, hue_deg=h_val, radius=radius)
    plot_annnotate_text_for_hue_deg(
        ax1, hue_deg=h_val, radius=radius, text=r"$h^{*}_{j}$",
        rate=text_rate, ha='left', va='center')

    line_dist_list = [15, 10, 5]
    idx_list_list = [[-1, 1, 0], [0, 2, 1], [1, 3, 2]]

    for line_dist, idx_list in zip(line_dist_list, idx_list_list):
        offset_x, offset_y = hue_chroma_to_ab(
            hue_deg=h_val+90, chroma=line_dist)
        plot_length_line_marker(
            ax1, aa_list, bb_list, idx_list, offset_x, offset_y)

    pu.show_and_save(
        fig=fig, legend_loc=None, show=False,
        save_fname=f"./img/medhod_a_{grid}x{grid}_h-{hue_idx}.png")


def cielab_method_b_ng_plot_cielab():
    l_val = 97
    cs_name = cs.BT709
    ab_max = 100
    ab_sample = 2048
    h_val = np.rad2deg(np.arctan2(90, -20)) + 0.2

    bl = 0.7 ** 2.4  # background luminance
    rgb_img = create_valid_cielab_ab_plane_image_gm24(
        l_val=l_val, ab_max=ab_max, ab_sample=ab_sample,
        color_space_name=cs_name,
        bg_rgb_luminance=np.array([bl, bl, bl]))

    title = f"a-b plane L*={l_val}, h*={h_val:.1f}°"

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=title,
        graph_title_size=None,
        xlabel="a", ylabel="b",
        axis_label_size=None,
        legend_size=17,
        xlim=[-60, 60],
        ylim=[-20, 100],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.grid(b=True, which='major', axis='both', color="#909090")
    ax1.imshow(
        rgb_img, extent=(-ab_max, ab_max, -ab_max, ab_max), aspect='auto')
    a_end, b_end = hue_chroma_to_ab(hue_deg=h_val, chroma=102)
    plot_annotate_line(
        ax1=ax1, st_pos=(0, 0), ed_pos=(a_end, b_end), color=(0.1, 0.1, 0.1),
        arrowstyle='-', linestyle='-')
    plot_annotate_line(
        ax1=ax1, st_pos=(0, 0), ed_pos=(60, 0), color=(0.1, 0.1, 0.1),
        arrowstyle='-', linestyle='-')
    radius = 6
    plot_arc_for_hue_deg(ax1=ax1, hue_deg=h_val, radius=radius)
    plot_annnotate_text_for_hue_deg(
        ax1, hue_deg=h_val, radius=radius, text=r"$h^{*}$",
        rate=1.03, ha='left', va='center')

    pu.show_and_save(
        fig=fig, legend_loc=None, show=False,
        save_fname="./img/method_b_cielab_ng.png")


def cielab_method_b_ng_plot_jzazbz(j_val=0.368):
    # j_val = 0.368
    cs_name = cs.BT709
    ab_max = 0.35
    ab_sample = 16384
    h_val = 252.8

    bl = 5000  # background luminance
    rgb_img = create_valid_jzazbz_ab_plane_image_gm24(
        j_val=j_val, ab_max=ab_max, ab_sample=ab_sample,
        color_space_name=cs_name,
        bg_rgb_luminance=np.array([bl, bl, bl]))

    title = f"a-b plane Jz={j_val}, hz={h_val:.1f}°"

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=title,
        graph_title_size=None,
        xlabel="a", ylabel="b",
        axis_label_size=None,
        legend_size=17,
        # xlim=[-0.15, 0.25],
        # ylim=[-0.35, 0.05],
        xlim=[-0.12, -0.05],
        ylim=[-0.32, -0.25],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.grid(b=True, which='major', axis='both', color="#909090")
    ax1.imshow(
        rgb_img, extent=(-ab_max, ab_max, -ab_max, ab_max), aspect='auto')
    a_end, b_end = hue_chroma_to_ab(hue_deg=h_val, chroma=0.33)
    a_st, b_st = hue_chroma_to_ab(hue_deg=h_val, chroma=0.265)
    plot_annotate_line(
        ax1=ax1, st_pos=(a_st, b_st), ed_pos=(a_end, b_end),
        color=(1.0, 0.3, 0.3), arrowstyle='-', linestyle='-')
    plot_annotate_line(
        ax1=ax1, st_pos=(0, 0), ed_pos=(0.25, 0), color=(0.1, 0.1, 0.1),
        arrowstyle='-', linestyle='-')
    radius = 0.03
    plot_arc_for_hue_deg(ax1=ax1, hue_deg=h_val, radius=radius)
    plot_annnotate_text_for_hue_deg(
        ax1, hue_deg=h_val, radius=radius, text=r"$h^{*}$",
        rate=1.03, ha='left', va='center')

    fname = f"./img/method_b_jzazbz_ng_{j_val:.3f}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)


def plot_method_c_part1(grid=8, hue_idx=1):
    hue_sample = grid
    method_a_num = 8
    chroma_init = 100
    c0 = chroma_init / (method_a_num - 1)
    h_val = np.linspace(0, 360, hue_sample)[hue_idx]
    x_max = int(chroma_init * np.cos(np.deg2rad(h_val)) * 1.1)
    y_max = int(chroma_init * np.sin(np.deg2rad(h_val)) * 1.1)
    xy_max = max([x_max, y_max])
    l_val_dummy = 70
    cs_name = cs.BT709
    ops_bak = np.get_printoptions()
    np.set_printoptions(precision=1)
    np.set_printoptions(**ops_bak)

    l_val = l_val_dummy

    ab_max = 80
    ab_sample = 1024

    bl = 0.96 ** 2.4  # background luminance
    rgb_img = create_valid_cielab_ab_plane_image_gm24(
        l_val=l_val, ab_max=ab_max, ab_sample=ab_sample,
        color_space_name=cs_name,
        bg_rgb_luminance=np.array([bl, bl, bl]))

    cc_list = np.linspace(0, chroma_init, method_a_num)
    aa_list, bb_list = hue_chroma_to_ab(hue_deg=h_val, chroma=cc_list)

    cc_list_a = calc_chroma_candidate_list_method_c(
        r_val_init=c0*4, c0=c0, lightness=l_val,
        hue_sample=hue_sample, cs_name=cs.BT709)[..., hue_idx]
    aa_a_list, bb_a_list = hue_chroma_to_ab(hue_deg=h_val, chroma=cc_list_a)

    title = f"a-b plane (L*={l_val:.2f}, Gamut={cs_name})"

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=title,
        graph_title_size=None,
        xlabel="a", ylabel="b",
        axis_label_size=None,
        legend_size=17,
        xlim=[-80, xy_max],
        ylim=[-80, xy_max],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        rgb_img, extent=(-ab_max, ab_max, -ab_max, ab_max), aspect='auto')

    sms = 12
    text_rate = 1.03
    ax1.plot(aa_list, bb_list, 'ks', ms=sms, mfc='w', zorder=5)

    ax1.plot(
        aa_list[4], bb_list[4], 'ks', ms=sms, mfc=pu.RED, zorder=5)

    plot_annotate_line(
        ax1=ax1, st_pos=(0, 0), ed_pos=(xy_max, 0), color=(0.1, 0.1, 0.1),
        arrowstyle='-')

    plot_annotate_line(
        ax1=ax1, st_pos=(0, 0), ed_pos=(aa_list[-1], bb_list[-1]),
        color=(0.1, 0.1, 0.1), arrowstyle='-')

    plot_annotate_text_only(
        ax1=ax1, pos=(aa_list[-1], bb_list[-1]), rate=text_rate,
        text=r"$Q_{j, n=N-1}$", ha='right', va='center')
    plot_annotate_text_only(
        ax1=ax1, pos=(aa_list[0], bb_list[0]), rate=text_rate,
        text=r"$Q_{j, n=0}$", ha='right', va='center')
    plot_annotate_text_only(
        ax1=ax1, pos=(aa_list[1], bb_list[1]), rate=text_rate,
        text=r"$Q_{j, n=1}$", ha='right', va='center')
    plot_annotate_text_only(
        ax1=ax1, pos=(aa_list[4], bb_list[4]), rate=text_rate,
        text=r"$Q_{j, n=4}$", ha='right', va='center')

    radius = 10
    plot_arc_for_hue_deg(ax1=ax1, hue_deg=h_val, radius=radius)
    plot_annnotate_text_for_hue_deg(
        ax1, hue_deg=h_val, radius=radius, text=r"$h^{*}_{j}$",
        rate=text_rate, ha='left', va='center')

    rot90_len = 20
    rot90_rad = np.deg2rad(h_val - 90)
    rot90_ofst_a = rot90_len * np.cos(rot90_rad)
    rot90_ofst_b = rot90_len * np.sin(rot90_rad)
    plot_annotate_line(
        ax1=ax1, st_pos=(0, 0), ed_pos=(rot90_ofst_a, rot90_ofst_b),
        color=(0.1, 0.1, 0.1), arrowstyle='-', linestyle='--')
    plot_annotate_line(
        ax1=ax1,
        st_pos=(aa_list[-1], bb_list[-1]),
        ed_pos=(aa_list[-1] + rot90_ofst_a, bb_list[-1]+rot90_ofst_b),
        color=(0.1, 0.1, 0.1), arrowstyle='-', linestyle='--')
    plot_annotate_line(
        ax1=ax1,
        st_pos=(rot90_ofst_a, rot90_ofst_b),
        ed_pos=(aa_list[-1] + rot90_ofst_a, bb_list[-1]+rot90_ofst_b),
        color=(0.1, 0.1, 0.1), arrowstyle='<->', linestyle='-')
    t_st_a = rot90_ofst_a + aa_list[-1] / 2
    t_st_b = rot90_ofst_b + bb_list[-1] / 2
    plot_annotate_text_only(
        ax1=ax1, pos=(t_st_a, t_st_b), rate=text_rate,
        text=r"$Q^{*}_{Q, j}$", ha='left', va='center')

    pu.show_and_save(
        fig=fig, legend_loc=None, show=False,
        save_fname="./img/medhod_c.png")


def plot_method_c_part2(grid=8, hue_idx=1):
    hue_sample = grid
    method_a_num = 8
    chroma_init = 100
    c0 = chroma_init / (method_a_num - 1)
    h_val = np.linspace(0, 360, hue_sample)[hue_idx]
    x_max = int(chroma_init * np.cos(np.deg2rad(h_val)) * 1.1)
    y_max = int(chroma_init * np.sin(np.deg2rad(h_val)) * 1.1)
    xy_max = max([x_max, y_max])
    l_val_dummy = 70
    cs_name = cs.BT709
    ops_bak = np.get_printoptions()
    np.set_printoptions(precision=1)
    np.set_printoptions(**ops_bak)

    l_val = l_val_dummy

    ab_max = 80
    ab_sample = 1024

    bl = 0.96 ** 2.4  # background luminance
    rgb_img = create_valid_cielab_ab_plane_image_gm24(
        l_val=l_val, ab_max=ab_max, ab_sample=ab_sample,
        color_space_name=cs_name,
        bg_rgb_luminance=np.array([bl, bl, bl]))

    cc_list = np.linspace(0, chroma_init, method_a_num)
    aa_list, bb_list = hue_chroma_to_ab(hue_deg=h_val, chroma=cc_list)

    cc_list_a = calc_chroma_candidate_list_method_c(
        r_val_init=c0*4, c0=c0, lightness=l_val,
        hue_sample=hue_sample, cs_name=cs.BT709)[..., hue_idx]
    aa_a_list, bb_a_list = hue_chroma_to_ab(hue_deg=h_val, chroma=cc_list_a)

    title = f"a-b plane (L*={l_val:.2f}, Gamut={cs_name})"

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=title,
        graph_title_size=None,
        xlabel="a", ylabel="b",
        axis_label_size=None,
        legend_size=17,
        xlim=[-10, xy_max],
        ylim=[-10, xy_max],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        rgb_img, extent=(-ab_max, ab_max, -ab_max, ab_max), aspect='auto')

    oms = 8
    sms = 12
    text_rate = 1.03
    ax1.plot(aa_list, bb_list, 'ks', ms=sms, mfc='w', zorder=5)
    ax1.plot(
        aa_list[4], bb_list[4], 'ks', ms=sms, mfc=pu.RED, zorder=5)
    ax1.plot(aa_a_list[0], bb_a_list[0], 'ko', ms=oms, mfc='k', zorder=5)
    ax1.plot(aa_a_list[1], bb_a_list[1], 'ko', ms=oms, mfc='k', zorder=5)
    ax1.plot(aa_a_list[2], bb_a_list[2], 'ko', ms=oms, mfc='k', zorder=5)

    plot_annotate_line(
        ax1=ax1, st_pos=(0, 0), ed_pos=(xy_max, 0), color=(0.1, 0.1, 0.1),
        arrowstyle='-')

    plot_annotate_line(
        ax1=ax1, st_pos=(0, 0), ed_pos=(aa_list[-1], bb_list[-1]),
        color=(0.1, 0.1, 0.1), arrowstyle='-')

    plot_annotate_text_only(
        ax1=ax1, pos=(aa_list[-1], bb_list[-1]), rate=text_rate,
        text=r"$Q_{j, n=N-1}$", ha='left', va='center')
    plot_annotate_text_only(
        ax1=ax1, pos=(aa_a_list[0], bb_a_list[0]), rate=text_rate,
        text=r"$P_{j,k=0}$", ha='left', va='center')
    plot_annotate_text_only(
        ax1=ax1, pos=(aa_a_list[1], bb_a_list[1]), rate=text_rate,
        text=r"$P_{j,k=1}$", ha='left', va='center')
    plot_annotate_text_only(
        ax1=ax1, pos=(aa_a_list[2], bb_a_list[2]), rate=text_rate,
        text=r"$P_{j,k=2}$", ha='left', va='center')
    # plot_annotate_text_only(
    #     ax1=ax1, pos=(aa_list[2], bb_list[2]), rate=text_rate,
    #     text=r"$P_{j,k+2}$", ha='left', va='center')
    # plot_annotate_text_only(
    #     ax1=ax1, pos=(aa_list[3], bb_list[3]), rate=text_rate,
    #     text=r"$P_{j,k+3}$", ha='left', va='center')

    radius = 10
    plot_arc_for_hue_deg(ax1=ax1, hue_deg=h_val, radius=radius)
    plot_annnotate_text_for_hue_deg(
        ax1, hue_deg=h_val, radius=radius, text=r"$h^{*}_{j}$",
        rate=text_rate, ha='left', va='center')

    line_dist_list = [5]
    idx_list_list = [[0, 2, 1]]

    for line_dist, idx_list in zip(line_dist_list, idx_list_list):
        offset_x, offset_y = hue_chroma_to_ab(
            hue_deg=h_val+90, chroma=line_dist)
        plot_length_line_marker(
            ax1, aa_a_list, bb_a_list, idx_list, offset_x, offset_y)

    pu.show_and_save(
        fig=fig, legend_loc=None, show=False,
        save_fname="./img/medhod_c_after.png")


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
    # plot_gamut_boundary_jzazbz()
    # plot_color_volume_cielab_rough_core(grid=8, idx=0)
    # plot_color_volume_cielab_rough_core(grid=16, idx=0)
    # plot_color_volume_cielab_rough_core(grid=32, idx=0)
    # plot_color_volume_cielab_rough_core(grid=64, idx=0)
    # plot_color_volume_cielab_rough(grid=8)
    # plot_color_volume_cielab_rough(grid=16)
    # plot_color_volume_cielab_rough(grid=32)
    # plot_color_volume_cielab_rough(grid=64)
    # plot_method_a(grid=8, hue_idx=1)
    # r_temp = calc_chroma_candidate_list(
    #     r_val_init=160, lightness=71.4, hue_sample=8, cs_name=cs.BT709)
    # print(r_temp[..., 1])
    # cielab_method_b_ng_plot_cielab()
    # cielab_method_b_ng_plot_jzazbz(j_val=0.362)
    # for idx in range(10):
    #     j_val = 0.36 + idx / 1000
    #     cielab_method_b_ng_plot_jzazbz(j_val=j_val)
    plot_method_c_part1(grid=8, hue_idx=1)
    # plot_method_c_part2(grid=8, hue_idx=1)
