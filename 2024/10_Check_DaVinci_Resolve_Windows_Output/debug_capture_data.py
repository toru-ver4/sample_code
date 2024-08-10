# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from pathlib import Path
from collections import namedtuple

# import third-party libraries
import numpy as np
from colour import XYZ_to_xy

# import my libraries
from jpeg_xr_decode import read_jpegxr
import plot_utility as pu
import transfer_functions as tf
import color_space as cs
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


SCRGB_TO_LUMINANCE_COEF = 80


def get_rgb_from_pos_list(img: np.ndarray, pos_list: np.ndarray):
    x_coords = pos_list[:, 0]
    y_coords = pos_list[:, 1]
    rgb = img[y_coords, x_coords]

    return rgb


def get_ramp_coordinate() -> np.ndarray:
    st_pos_h = 115
    ed_pos_h = 2163
    st_pos_v = 590
    pos_h = np.arange(st_pos_h, ed_pos_h, 2)
    pos_v = np.ones_like(pos_h) * st_pos_v
    pos = np.column_stack([pos_h, pos_v])

    return pos


def get_colorchecker_coordinate() -> np.ndarray:
    st_pos = (2380, 328)
    ed_pos = (3596, 1060)
    num_of_v = 4
    num_of_h = 6
    pos_h = np.round(np.linspace(st_pos[0], ed_pos[0], num_of_h))\
        .astype(np.uint16)
    pos_v = np.round(np.linspace(st_pos[1], ed_pos[1], num_of_v))\
        .astype(np.uint16)
    pos_hh, pos_vv = np.meshgrid(pos_h, pos_v)
    pos = np.dstack([pos_hh, pos_vv]).reshape(-1, 2)

    return pos


def plot_sdr_content_bright_result():
    PlotParam = namedtuple(
        'PlotParam', [
            'img_fname', 'plot_color', 'brght_val', "lw"
        ])
    plot_param_list = [
        PlotParam(
            "./debug/jpeg_xr/No1_SDR_bright_000.jxr", pu.RED, 0, 8),
        PlotParam(
            "./debug/jpeg_xr/No1_SDR_bright_050.jxr", pu.GREEN, 50, 5),
        PlotParam(
            "./debug/jpeg_xr/No1_SDR_bright_100.jxr", pu.BLUE, 100, 2),
    ]

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=None,
        graph_title_size=None,
        xlabel="Target Luminance (nits)",
        ylabel="Captured Lumiannce (nits)",
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
    x = np.linspace(0, 1, 1024)
    ref_y = tf.eotf_to_luminance(x, tf.ST2084)

    for plot_param in plot_param_list:
        img = read_jpegxr(plot_param.img_fname)
        pos_list = get_ramp_coordinate()
        ramp_rgb = get_rgb_from_pos_list(img=img, pos_list=pos_list)
        ramp_y = ramp_rgb[..., 1] * SCRGB_TO_LUMINANCE_COEF
        ax1.plot(
            ref_y, ramp_y,
            color=plot_param.plot_color,
            lw=plot_param.lw,
            label=f"SDR content brightness = {plot_param.brght_val}")

    ax1.plot(ref_y, ref_y, '--', color='k', label="Reference")

    save_fname = "./img/sdr_content_bight_test_result.png"
    print(save_fname)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', show=True, save_fname=save_fname)


def debug_limited_full():
    PlotParam = namedtuple(
        'PlotParam', ['img_fname', 'plot_color', 'label', 'lw'])
    plot_param_list = [
        PlotParam(
            "./debug/jpeg_xr/No1_SDR_bright_000.jxr",
            pu.RED,
            "Video",
            6
        ),
        PlotParam(
            "./debug/jpeg_xr/No1_SDR_bright_000_full_range.jxr",
            pu.BLUE,
            "Full",
            2
        ),
    ]

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=None,
        graph_title_size=None,
        xlabel="Target Luminance (nits)",
        ylabel="Captured Lumiannce (nits)",
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
    x = np.linspace(0, 1, 1024)
    ref_y = tf.eotf_to_luminance(x, tf.ST2084)

    for plot_param in plot_param_list:
        img = read_jpegxr(plot_param.img_fname)
        pos_list = get_ramp_coordinate()
        ramp_rgb = get_rgb_from_pos_list(img=img, pos_list=pos_list)
        ramp_y = ramp_rgb[..., 1] * SCRGB_TO_LUMINANCE_COEF
        ax1.plot(
            ref_y, ramp_y,
            color=plot_param.plot_color,
            label=f"{plot_param.label}",
            lw=plot_param.lw
        )

    # ax1.plot(ref_y, ref_y, '--', color='k', label="Reference")

    save_fname = "./debug/img/debug_limited_full.png"
    print(save_fname)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', show=True, save_fname=save_fname)


def plot_mhc_profile_luminance_result():
    PlotParam = namedtuple(
        'PlotParam', [
            'img_fname', 'plot_color', 'profile', "lw"
        ])
    plot_param_list = [
        PlotParam(
            "./debug/jpeg_xr/No2_MHC_10000nits_BT2020.jxr",
            pu.RED,
            "10000 nits-Rec.2020",
            5
        ),
        PlotParam(
            "./debug/jpeg_xr/No2_MHC_400nits_BT709.jxr",
            pu.BLUE,
            "400 nits-Rec.709",
            2
        ),
    ]

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=None,
        graph_title_size=None,
        xlabel="Target Luminance (nits)",
        ylabel="Captured Lumiannce (nits)",
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
    x = np.linspace(0, 1, 1024)
    ref_y = tf.eotf_to_luminance(x, tf.ST2084)

    for plot_param in plot_param_list:
        img = read_jpegxr(plot_param.img_fname)
        pos_list = get_ramp_coordinate()
        ramp_rgb = get_rgb_from_pos_list(img=img, pos_list=pos_list)
        ramp_y = ramp_rgb[..., 1] * SCRGB_TO_LUMINANCE_COEF
        ax1.plot(
            ref_y, ramp_y,
            color=plot_param.plot_color,
            lw=plot_param.lw,
            label=f"MHC Profile: {plot_param.profile}")

    ax1.plot(ref_y, ref_y, '--', color='k', label="Reference")

    save_fname = "./img/MHC_Profile_Result.png"
    print(save_fname)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', show=True, save_fname=save_fname)


def plot_mhc_profile_chromaticity_diagram():
    PlotParam = namedtuple(
        'PlotParam', [
            'img_fname', 'plot_color', 'profile', "marker", 'facecolor',
            'edgecolor', 'color'
        ])
    plot_param_list = [
        PlotParam(
            "./debug/jpeg_xr/No2_MHC_10000nits_BT2020.jxr",
            pu.RED,
            "MHC Profile: 10000 nits-Rec.2020",
            "o",
            None,
            'k',
            'k'
        ),
        PlotParam(
            "./debug/jpeg_xr/No2_MHC_400nits_BT709.jxr",
            pu.BLUE,
            "MHC Profile: 400 nits-Rec.709",
            "+",
            "none",
            None,
            pu.RED
        ),
    ]

    ref_xy = tpg.generate_color_checker_xyY_value()[:18]

    rate = 1.0
    xmin = -0.1
    xmax = 0.8
    ymin = -0.1
    ymax = 0.9
    st_wl = 380
    ed_wl = 780
    wl_step = 1
    plot_wl_list = [
        410, 450, 470, 480, 485, 490, 495,
        500, 505, 510, 520, 530, 540, 550, 560, 570, 580, 590,
        600, 620, 690]
    cmf_xy = pu.calc_horseshoe_chromaticity(
        st_wl=st_wl, ed_wl=ed_wl, wl_step=wl_step)
    cmf_xy_norm = pu.calc_normal_pos(
        xy=cmf_xy, normal_len=0.05, angle_degree=90)
    wl_list = np.arange(st_wl, ed_wl + 1, wl_step)
    xy_image = pu.get_chromaticity_image(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, cmf_xy=cmf_xy)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20 * rate,
        figsize=((xmax - xmin) * 10 * rate,
                 (ymax - ymin) * 10 * rate),
        graph_title="Chromaticity Gamut",
        graph_title_size=None,
        xlabel=None, ylabel=None,
        axis_label_size=None,
        legend_size=14 * rate,
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
        xtick=[x * 0.1 + xmin for x in
               range(int((xmax - xmin)/0.1) + 1)],
        ytick=[x * 0.1 + ymin for x in
               range(int((ymax - ymin)/0.1) + 1)],
        xtick_size=17 * rate,
        ytick_size=17 * rate,
        linewidth=4 * rate,
        minor_xtick_num=2,
        minor_ytick_num=2)
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=2*rate, label=None)
    for idx, wl in enumerate(wl_list):
        if wl not in plot_wl_list:
            continue
        pu.draw_wl_annotation(
            ax1=ax1, wl=wl, rate=rate,
            st_pos=[cmf_xy_norm[idx, 0], cmf_xy_norm[idx, 1]],
            ed_pos=[cmf_xy[idx, 0], cmf_xy[idx, 1]])
    bt709_gamut = pu.get_primaries(name=cs.BT709)
    ax1.plot(bt709_gamut[:, 0], bt709_gamut[:, 1], '-o',
             c=pu.RED, label="Rec.709 Chromaticity Gamut", lw=3*rate)
    ax1.scatter(
        ref_xy[..., 0], ref_xy[..., 1], marker='s', s=150*rate,
        facecolors='none', edgecolors='black', linewidths=2*rate,
        label="Reference")

    for plot_param in plot_param_list:
        img = read_jpegxr(plot_param.img_fname)
        pos_list = get_colorchecker_coordinate()
        cc_rgb = get_rgb_from_pos_list(img=img, pos_list=pos_list)
        cc_rgb = cc_rgb * SCRGB_TO_LUMINANCE_COEF / 100
        large_xyz = cs.rgb_to_large_xyz(rgb=cc_rgb, color_space_name=cs.BT709)
        xy = XYZ_to_xy(large_xyz)[:18]
        ax1.scatter(
            xy[..., 0], xy[..., 1],
            marker=plot_param.marker, s=70*rate,
            c=plot_param.color, facecolors=plot_param.facecolor,
            edgecolors=plot_param.edgecolor,
            linewidths=2*rate, label=plot_param.profile
        )

    ax1.plot(
        [0.3127], [0.3290], 'x', label='D65', ms=12*rate, mew=2*rate,
        color='k', alpha=0.8, zorder=100)
    ax1.imshow(xy_image, extent=(xmin, xmax, ymin, ymax), alpha=0.5)
    save_fname = "./img/MHC_Profile_Chromaticity_Diagram.png"
    pu.show_and_save(
        fig=fig, legend_loc='upper right', show=True, save_fname=save_fname)


def plot_rec709_gm24_luminance_result():
    PlotParam = namedtuple(
        'PlotParam', [
            'img_fname', 'plot_color', 'profile', "lw"
        ])
    plot_param_list = [
        PlotParam(
            "./debug/jpeg_xr/No4_Rec709_Gamma2.4.jxr",
            pu.RED,
            "Rec.709-Gamma2.4",
            3
        ),
    ]

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=None,
        graph_title_size=None,
        xlabel="Target Luminance (nits)",
        ylabel="Captured Lumiannce (nits)",
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
    x = np.linspace(0, 1, 1024)
    ref_y = tf.eotf_to_luminance(x, tf.GAMMA24)

    for plot_param in plot_param_list:
        img = read_jpegxr(plot_param.img_fname)
        pos_list = get_ramp_coordinate()
        ramp_rgb = get_rgb_from_pos_list(img=img, pos_list=pos_list)
        ramp_y = ramp_rgb[..., 1] * SCRGB_TO_LUMINANCE_COEF
        ax1.plot(
            ref_y, ramp_y,
            color=plot_param.plot_color,
            lw=plot_param.lw,
            label=f"{plot_param.profile}"
        )

    ax1.plot(ref_y, ref_y, '--', color='k', label="Reference")

    save_fname = "./img/Rec709_gm24_Result.png"
    print(save_fname)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', show=True, save_fname=save_fname)


def plot_rec709_gm24_chromaticity_diagram():
    PlotParam = namedtuple(
        'PlotParam', [
            'img_fname', 'plot_color', 'profile', "marker", 'facecolor',
            'edgecolor', 'color'
        ])
    plot_param_list = [
        PlotParam(
            "./debug/jpeg_xr/No4_Rec709_Gamma2.4.jxr",
            pu.BLUE,
            "MHC Profile: 400 nits-Rec.709",
            "+",
            "none",
            None,
            pu.BLUE
        ),
    ]

    ref_xy = tpg.generate_color_checker_xyY_value()[:18]

    rate = 1.0
    xmin = -0.1
    xmax = 0.8
    ymin = -0.1
    ymax = 0.9
    st_wl = 380
    ed_wl = 780
    wl_step = 1
    plot_wl_list = [
        410, 450, 470, 480, 485, 490, 495,
        500, 505, 510, 520, 530, 540, 550, 560, 570, 580, 590,
        600, 620, 690]
    cmf_xy = pu.calc_horseshoe_chromaticity(
        st_wl=st_wl, ed_wl=ed_wl, wl_step=wl_step)
    cmf_xy_norm = pu.calc_normal_pos(
        xy=cmf_xy, normal_len=0.05, angle_degree=90)
    wl_list = np.arange(st_wl, ed_wl + 1, wl_step)
    xy_image = pu.get_chromaticity_image(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, cmf_xy=cmf_xy)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20 * rate,
        figsize=((xmax - xmin) * 10 * rate,
                 (ymax - ymin) * 10 * rate),
        graph_title="Chromaticity Gamut",
        graph_title_size=None,
        xlabel=None, ylabel=None,
        axis_label_size=None,
        legend_size=14 * rate,
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
        xtick=[x * 0.1 + xmin for x in
               range(int((xmax - xmin)/0.1) + 1)],
        ytick=[x * 0.1 + ymin for x in
               range(int((ymax - ymin)/0.1) + 1)],
        xtick_size=17 * rate,
        ytick_size=17 * rate,
        linewidth=4 * rate,
        minor_xtick_num=2,
        minor_ytick_num=2)
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=2*rate, label=None)
    for idx, wl in enumerate(wl_list):
        if wl not in plot_wl_list:
            continue
        pu.draw_wl_annotation(
            ax1=ax1, wl=wl, rate=rate,
            st_pos=[cmf_xy_norm[idx, 0], cmf_xy_norm[idx, 1]],
            ed_pos=[cmf_xy[idx, 0], cmf_xy[idx, 1]])
    bt709_gamut = pu.get_primaries(name=cs.BT709)
    ax1.plot(bt709_gamut[:, 0], bt709_gamut[:, 1], '-o',
             c=pu.RED, label="Rec.709 Chromaticity Gamut", lw=3*rate,
             zorder=50)
    ax1.scatter(
        ref_xy[..., 0], ref_xy[..., 1], marker='s', s=150*rate,
        facecolors='none', edgecolors='black', linewidths=2*rate,
        label="Reference")

    for plot_param in plot_param_list:
        img = read_jpegxr(plot_param.img_fname)
        pos_list = get_colorchecker_coordinate()
        cc_rgb = get_rgb_from_pos_list(img=img, pos_list=pos_list)
        cc_rgb = cc_rgb * SCRGB_TO_LUMINANCE_COEF / 100
        large_xyz = cs.rgb_to_large_xyz(rgb=cc_rgb, color_space_name=cs.BT709)
        xy = XYZ_to_xy(large_xyz)[:18]
        ax1.scatter(
            xy[..., 0], xy[..., 1],
            marker=plot_param.marker, s=70*rate,
            c=plot_param.color, facecolors=plot_param.facecolor,
            edgecolors=plot_param.edgecolor,
            linewidths=2*rate, label=plot_param.profile
        )

    ax1.plot(
        [0.3127], [0.3290], 'x', label='D65', ms=12*rate, mew=2*rate,
        color='k', alpha=0.8, zorder=100)
    ax1.imshow(xy_image, extent=(xmin, xmax, ymin, ymax), alpha=0.5)
    save_fname = "./img/Rec709_gm24_Chromaticity_Diagram.png"
    pu.show_and_save(
        fig=fig, legend_loc='upper right', show=True, save_fname=save_fname)


def plot_p3d65_st2084_luminance_result():
    PlotParam = namedtuple(
        'PlotParam', [
            'img_fname', 'plot_color', 'profile', "lw"
        ])
    plot_param_list = [
        PlotParam(
            "./debug/jpeg_xr/No3_P3D65_PQ.jxr",
            pu.RED,
            "P3D65-ST2084",
            2
        ),
    ]

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=None,
        graph_title_size=None,
        xlabel="Target Luminance (nits)",
        ylabel="Captured Lumiannce (nits)",
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
    x = np.linspace(0, 1, 1024)
    ref_y = tf.eotf_to_luminance(x, tf.ST2084)

    for plot_param in plot_param_list:
        img = read_jpegxr(plot_param.img_fname)
        pos_list = get_ramp_coordinate()
        ramp_rgb = get_rgb_from_pos_list(img=img, pos_list=pos_list)
        ramp_y = ramp_rgb[..., 1] * SCRGB_TO_LUMINANCE_COEF
        ax1.plot(
            ref_y, ramp_y,
            color=plot_param.plot_color,
            lw=plot_param.lw,
            label=f"{plot_param.profile}")

    ax1.plot(ref_y, ref_y, '--', color='k', label="Reference")

    save_fname = "./img/P3D65-ST2084_Result.png"
    print(save_fname)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', show=True, save_fname=save_fname)


def plot_chromaticity_diagram():
    PlotParam = namedtuple(
        'PlotParam', [
            'img_fname', 'plot_color', 'profile', "marker", 'facecolor',
            'edgecolor', 'color'
        ]
    )
    plot_param_list = [
        PlotParam(
            "./debug/jpeg_xr/No3_P3D65_PQ.jxr",
            pu.BLUE,
            "MHC Profile: 400 nits-Rec.709",
            "+",
            "none",
            None,
            pu.BLUE
        ),
    ]

    ref_xy = tpg.generate_color_checker_xyY_value()[:18]

    rate = 1.0
    xmin = -0.1
    xmax = 0.8
    ymin = -0.1
    ymax = 0.9
    st_wl = 380
    ed_wl = 780
    wl_step = 1
    plot_wl_list = [
        410, 450, 470, 480, 485, 490, 495,
        500, 505, 510, 520, 530, 540, 550, 560, 570, 580, 590,
        600, 620, 690]
    cmf_xy = pu.calc_horseshoe_chromaticity(
        st_wl=st_wl, ed_wl=ed_wl, wl_step=wl_step)
    cmf_xy_norm = pu.calc_normal_pos(
        xy=cmf_xy, normal_len=0.05, angle_degree=90)
    wl_list = np.arange(st_wl, ed_wl + 1, wl_step)
    xy_image = pu.get_chromaticity_image(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, cmf_xy=cmf_xy)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20 * rate,
        figsize=((xmax - xmin) * 10 * rate,
                 (ymax - ymin) * 10 * rate),
        graph_title="Chromaticity Gamut",
        graph_title_size=None,
        xlabel=None, ylabel=None,
        axis_label_size=None,
        legend_size=14 * rate,
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
        xtick=[x * 0.1 + xmin for x in
               range(int((xmax - xmin)/0.1) + 1)],
        ytick=[x * 0.1 + ymin for x in
               range(int((ymax - ymin)/0.1) + 1)],
        xtick_size=17 * rate,
        ytick_size=17 * rate,
        linewidth=4 * rate,
        minor_xtick_num=2,
        minor_ytick_num=2)
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=2*rate, label=None)
    for idx, wl in enumerate(wl_list):
        if wl not in plot_wl_list:
            continue
        pu.draw_wl_annotation(
            ax1=ax1, wl=wl, rate=rate,
            st_pos=[cmf_xy_norm[idx, 0], cmf_xy_norm[idx, 1]],
            ed_pos=[cmf_xy[idx, 0], cmf_xy[idx, 1]])
    bt709_gamut = pu.get_primaries(name=cs.BT709)
    ax1.plot(bt709_gamut[:, 0], bt709_gamut[:, 1], '-o',
             c=pu.RED, label="Rec.709 Chromaticity Gamut", lw=3*rate,
             zorder=50)
    ax1.scatter(
        ref_xy[..., 0], ref_xy[..., 1], marker='s', s=150*rate,
        facecolors='none', edgecolors='black', linewidths=2*rate,
        label="Reference")

    for plot_param in plot_param_list:
        img = read_jpegxr(plot_param.img_fname)
        pos_list = get_colorchecker_coordinate()
        cc_rgb = get_rgb_from_pos_list(img=img, pos_list=pos_list)
        cc_rgb = cc_rgb * SCRGB_TO_LUMINANCE_COEF / 100
        large_xyz = cs.rgb_to_large_xyz(rgb=cc_rgb, color_space_name=cs.BT709)
        xy = XYZ_to_xy(large_xyz)[:18]
        ax1.scatter(
            xy[..., 0], xy[..., 1],
            marker=plot_param.marker, s=70*rate,
            c=plot_param.color, facecolors=plot_param.facecolor,
            edgecolors=plot_param.edgecolor,
            linewidths=2*rate, label=plot_param.profile
        )

    ax1.plot(
        [0.3127], [0.3290], 'x', label='D65', ms=12*rate, mew=2*rate,
        color='k', alpha=0.8, zorder=100)
    ax1.imshow(xy_image, extent=(xmin, xmax, ymin, ymax), alpha=0.5)
    save_fname = "./img/Rec709_gm24_Chromaticity_Diagram.png"
    pu.show_and_save(
        fig=fig, legend_loc='upper right', show=True, save_fname=save_fname)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # debug_limited_full()
    # plot_sdr_content_bright_result()
    # plot_mhc_profile_luminance_result()
    # plot_mhc_profile_chromaticity_diagram()
    # plot_p3d65_st2084_luminance_result()
    # plot_rec709_gm24_luminance_result()
    # plot_rec709_gm24_chromaticity_diagram()
