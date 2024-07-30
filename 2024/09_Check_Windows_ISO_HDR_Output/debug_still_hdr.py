# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
import re

# import third-party libraries
import numpy as np
from colour.io import read_image

# import my libraries
from test_pattern_coordinate import GridCoordinate, ImgWithTextCoordinate
import test_pattern_generator2 as tpg
import transfer_functions as tf
import font_control2 as fc2
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def carete_sdr_composite_tp(sdr_ref_white=203, hdr_ref_white=203):
    patch_luminance_lust = [
        60, 80, 100, 120, 140, 160, 180, 203, 220, 240, 260, 280, 300
    ]
    max_luminance = 300
    white_rate = sdr_ref_white / hdr_ref_white
    width = 1920
    height = 200
    patch_size = 100
    dummy_text = "203 nits"
    font_size = 28
    font_path = fc2.NOTO_SANS_MONO_BOLD
    out_fname = f"./img/exp_sdrw-{sdr_ref_white}_hdrw-{hdr_ref_white}.png"
    gc = GridCoordinate(
        bg_width=width, bg_height=height,
        fg_width=patch_size, fg_height=patch_size,
        h_num=len(patch_luminance_lust), v_num=1
    )
    pos_list = gc.get_st_pos_list()[:, 0, :]

    img = np.ones((height, width, 3)) * hdr_ref_white
    tpg.draw_outline(img, fg_color=(0, 0, 0), outline_width=1)
    for idx, pos in enumerate(pos_list):
        patch_img = np.ones((patch_size, patch_size, 3))
        patch_img = patch_img * patch_luminance_lust[idx]
        tpg.merge(img, patch_img, pos=[pos[0], pos[1]])

        img_text_coorinate = ImgWithTextCoordinate(
            img_width=patch_size,
            img_height=patch_size,
            text=dummy_text,
            font_size=font_size,
            font_path=font_path,
            text_pos="bottom",
            margin_num_of_chara=0.5
        )
        _, text_st_pos = img_text_coorinate.get_img_and_text_st_pos()
        text_st_pos = text_st_pos + pos
        text_draw_ctrl = fc2.TextDrawControl(
            text=f"{patch_luminance_lust[idx]} nits",
            font_color=(0, 0, 0),
            font_size=font_size,
            font_path=font_path
        )
        text_draw_ctrl.draw(img=img, pos=text_st_pos)
    img = img * white_rate

    img_out = tf.oetf(
        img/max_luminance, tf.GAMMA24)

    print(out_fname)
    tpg.img_wirte_float_as_16bit_int(out_fname, img_out)


def get_ramp_data_from_avif_tp(fname):
    img = read_image(fname)[..., :3]
    pos_list_h = np.round(np.linspace(216, 1620, 65)).astype(np.uint16)
    pos_list_v = 1466
    ramp_data = img[pos_list_v, pos_list_h]

    return ramp_data


def plot_avif_tp_tf():
    tp_fname_list = [
        "./debug/jpeg_xr_to_exr/TP_480_nits.exr",
        "./debug/jpeg_xr_to_exr/TP_280_nits.exr",
        "./debug/jpeg_xr_to_exr/TP_204_nits.exr",
        # "./debug/jpeg_xr_to_exr/TP_200_nits.exr",
        "./debug/jpeg_xr_to_exr/TP_140_nits.exr",
        "./debug/jpeg_xr_to_exr/TP_080_nits.exr",
    ]
    pattern = re.compile(r'TP_(\d+)_nits\.exr')
    nits_list = [
        f"SDR Ref White = {int(pattern.search(fname).group(1))} nits"
        for fname in tp_fname_list
    ]
    x_cv = np.linspace(0, 1, 65)
    x_nits = tf.eotf_to_luminance(x_cv, tf.ST2084)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Relationship Between SDR Ref White and Display Luminance",
        graph_title_size=20,
        xlabel="Target Luminance (nits)",
        ylabel="Measured Lumiannce (nits)",
        axis_label_size=None,
        legend_size=16,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=4,
        minor_xtick_num=None,
        minor_ytick_num=None)
    # pu.log_scale_settings(ax1=ax1)
    for idx in range(len(tp_fname_list)):
        tp_fname = tp_fname_list[idx]
        y = get_ramp_data_from_avif_tp(fname=tp_fname)[..., 1]
        y_nits = y * 80
        label = nits_list[idx]
        ax1.plot(x_nits, y_nits, label=label)
    ax1.plot(x_nits, x_nits, '--', color="k", label="Reference", lw=3)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', show=False,
        save_fname="./debug/plot/tp_tf.png")

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Relationship Between SDR Ref White and Display Luminance",
        graph_title_size=20,
        xlabel="Target Luminance (nits)",
        ylabel="Measured Lumiannce (nits)",
        axis_label_size=None,
        legend_size=16,
        xlim=[4000, 6000],
        ylim=[4000, 6000],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    # pu.log_scale_settings(ax1=ax1)
    for idx in [2, 3]:
        tp_fname = tp_fname_list[idx]
        y = get_ramp_data_from_avif_tp(fname=tp_fname)[..., 1]
        y_nits = y * 80
        label = nits_list[idx]
        ax1.plot(x_nits, y_nits, label=label)
    ax1.plot(x_nits, x_nits, '--', color="k", label="Reference")
    pu.show_and_save(
        fig=fig, legend_loc='upper left', show=False,
        save_fname="./debug/plot/tp_tf_crop.png")


def plot_blog_avif_characteristics():
    tp_fname_list = [
        "./debug/jpeg_xr_to_exr/TP_480_nits.exr",
        "./debug/jpeg_xr_to_exr/TP_280_nits.exr",
        # "./debug/jpeg_xr_to_exr/TP_204_nits.exr",
        # "./debug/jpeg_xr_to_exr/TP_200_nits.exr",
        "./debug/jpeg_xr_to_exr/TP_140_nits.exr",
        "./debug/jpeg_xr_to_exr/TP_080_nits.exr",
    ]
    pattern = re.compile(r'TP_(\d+)_nits\.exr')
    nits_list = [
        f"SDR content brightness = {(int(pattern.search(fname).group(1))-80)//4}"
        for fname in tp_fname_list
    ]
    x_cv = np.linspace(0, 1, 65)
    x_nits = tf.eotf_to_luminance(x_cv, tf.ST2084)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Relationship Between SDR Content Brightness and Display Luminance",
        graph_title_size=18,
        xlabel="Target Luminance (nits)",
        ylabel="Measured Lumiannce (nits)",
        axis_label_size=None,
        legend_size=16,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=4,
        minor_xtick_num=None,
        minor_ytick_num=None)
    # pu.log_scale_settings(ax1=ax1)
    for idx in range(len(tp_fname_list)):
        tp_fname = tp_fname_list[idx]
        y = get_ramp_data_from_avif_tp(fname=tp_fname)[..., 1]
        y_nits = y * 80
        label = nits_list[idx]
        ax1.plot(x_nits, y_nits, label=label)
    ax1.plot(x_nits, x_nits, '--', color="k", label="Reference", lw=3)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', show=True,
        save_fname="./img/avif_edge_tonemap.png")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # sdr_hdr_white_pair = [
    #     [80, 100], [100, 100],
    #     [80, 203], [100, 203], [203, 203]
    # ]
    # for sdr_ref_white, hdr_ref_white in sdr_hdr_white_pair:
    #     carete_sdr_composite_tp(
    #         sdr_ref_white=sdr_ref_white, hdr_ref_white=hdr_ref_white)

    # plot_avif_tp_tf()
    plot_blog_avif_characteristics()
