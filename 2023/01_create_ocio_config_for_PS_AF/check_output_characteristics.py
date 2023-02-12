# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour.io import write_image, read_image

# import my libraries
import test_pattern_generator2 as tpg
import transfer_functions as tf
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def create_pq_based_ramp_exr():
    width = 1920
    height = 1080
    out_fname = "./img/pq_based_ramp_linear.exr"
    x = np.linspace(0, 1, width)
    linear = tf.eotf_to_luminance(x, tf.ST2084) / 100
    img = tpg.h_mono_line_to_img(linear, height)

    write_image(img, out_fname)


def plot_Ae_HDR10_out_with_white(graph_name="./img/out_w203.png"):
    width = 1920

    fname_list = [
        "./img/Ae_out_Chara/Ae_out_chara_ProRes_W-none.tif",
        "./img/Ae_out_Chara/Ae_out_chara_ProRes_W-100.tif",
        "./img/Ae_out_Chara/Ae_out_chara_ProRes_W-203.tif",
        "./img/Ae_out_Chara/Ae_out_chara_ProRes_W-10000.tif"]
    profile_list = [
        "Rec.2100 PQ", "Rec.2100 PQ W100", "Rec.2100 PQ W203",
        "Rec.2100 PQ W10000"]

    def get_linear(fname):
        img = tpg.img_read_as_float(fname)
        x_ae = img[0, :, 1]  # extract Green line data
        ae_luminance = tf.eotf_to_luminance(x_ae, tf.ST2084)
        return ae_luminance

    chara_list = [get_linear(fname) for fname in fname_list]

    x = np.linspace(0, 1, width)
    ref_luminance = tf.eotf_to_luminance(x, tf.ST2084)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Ae In-Out Characteristics (ProRes 4444)",
        graph_title_size=None,
        xlabel="Input Luminance [cd/m2]",
        ylabel="Output LUminance [cd/m2]",
        axis_label_size=None,
        legend_size=17,
        xlim=[0.009, 15000],
        ylim=[0.009, 15000],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    pu.log_scale_settings(ax1=ax1)
    for idx, y in enumerate(chara_list):
        ax1.plot(ref_luminance, y, label=profile_list[idx])
    pu.show_and_save(
        fig=fig, legend_loc='upper left', save_fname=graph_name,
        show=False)


def plot_Ae_HDR10_exr_out_with_white(graph_name="./img/out.png"):
    width = 1920

    fname_list = [
        "./AfterEffects/Comp 1/Comp 1_W-none_00000.exr",
        "./AfterEffects/Comp 1/Comp 1_W-100_00000.exr",
        "./AfterEffects/Comp 1/Comp 1_W-203_00000.exr",
        "./AfterEffects/Comp 1/Comp 1_W-10000_00000.exr"]
    profile_list = [
        "Rec.2100 PQ", "Rec.2100 PQ W100", "Rec.2100 PQ W203",
        "Rec.2100 PQ W10000"]

    def get_linear(fname):
        img = read_image(fname)
        x_ae = img[0, :, 1]  # extract Green line data
        ae_luminance = x_ae * 100
        return ae_luminance

    chara_list = [get_linear(fname) for fname in fname_list]

    x = np.linspace(0, 1, width)
    ref_luminance = tf.eotf_to_luminance(x, tf.ST2084)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Ae In-Out Characteristics (EXR Linear)",
        graph_title_size=None,
        xlabel="Input Luminance [cd/m2]",
        ylabel="Output LUminance [cd/m2]",
        axis_label_size=None,
        legend_size=17,
        xlim=[0.009, 15000],
        ylim=[0.009, 15000],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    pu.log_scale_settings(ax1=ax1)
    for idx, y in enumerate(chara_list):
        ax1.plot(ref_luminance, y, label=profile_list[idx])
    pu.show_and_save(
        fig=fig, legend_loc='upper left', save_fname=graph_name,
        show=False)


def plot_Ae_HDR10_exr_nl_out_with_white(graph_name="./img/out.png"):
    width = 1920

    fname_list = [
        "./AfterEffects/Comp 1/Comp 1_NL_W-none_00000.exr",
        "./AfterEffects/Comp 1/Comp 1_NL_W-100_00000.exr",
        "./AfterEffects/Comp 1/Comp 1_NL_W-203_00000.exr",
        "./AfterEffects/Comp 1/Comp 1_NL_W-10000_00000.exr"]
    profile_list = [
        "Rec.2100 PQ", "Rec.2100 PQ W100", "Rec.2100 PQ W203",
        "Rec.2100 PQ W10000"]

    def get_linear(fname):
        img = read_image(fname)
        x_ae = img[0, :, 1]  # extract Green line data
        ae_luminance = tf.eotf_to_luminance(x_ae, tf.ST2084)
        return ae_luminance

    chara_list = [get_linear(fname) for fname in fname_list]

    x = np.linspace(0, 1, width)
    ref_luminance = tf.eotf_to_luminance(x, tf.ST2084)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Ae In-Out Characteristics (EXR Non-linear)",
        graph_title_size=None,
        xlabel="Input Luminance [cd/m2]",
        ylabel="Output LUminance [cd/m2]",
        axis_label_size=None,
        legend_size=17,
        xlim=[0.009, 15000],
        ylim=[0.009, 15000],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    pu.log_scale_settings(ax1=ax1)
    for idx, y in enumerate(chara_list):
        ax1.plot(ref_luminance, y, label=profile_list[idx])
    pu.show_and_save(
        fig=fig, legend_loc='upper left', save_fname=graph_name,
        show=False)


def plot_Ae_HDR10_exr_nl_out_vari_prj_white(graph_name="./img/out.png"):
    """
    Plot transfer characteristics of the working color space.
    Output color space is fixed to "Rec.2100 PQ" (with no white point).
    """
    width = 1920

    fname_list = [
        "./AfterEffects/Comp 1/prj_w_W-none _00000.exr",
        "./AfterEffects/Comp 1/prj_w_W-100 _00000.exr",
        "./AfterEffects/Comp 1/prj_w_W-203 _00000.exr",
        "./AfterEffects/Comp 1/prj_w_W-10000 _00000.exr",
    ]
    profile_list = [
        "Rec.2100 PQ",
        "Rec.2100 PQ W100",
        "Rec.2100 PQ W203",
        "Rec.2100 PQ W10000"
    ]

    def get_linear(fname):
        img = read_image(fname)
        x_ae = img[0, :, 1]  # extract Green line data
        ae_luminance = tf.eotf_to_luminance(x_ae, tf.ST2084)
        return ae_luminance

    chara_list = [get_linear(fname) for fname in fname_list]

    x = np.linspace(0, 1, width)
    ref_luminance = tf.eotf_to_luminance(x, tf.ST2084)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Transfer characteristics of the working color space",
        graph_title_size=None,
        xlabel="Input Luminance [cd/m2]",
        ylabel="Output LUminance [cd/m2]",
        axis_label_size=None,
        legend_size=17,
        xlim=[0.009, 15000],
        ylim=[0.009, 15000],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    pu.log_scale_settings(ax1=ax1)
    for idx, y in enumerate(chara_list):
        ax1.plot(ref_luminance, y, label=profile_list[idx])
    pu.show_and_save(
        fig=fig, legend_loc='upper left', save_fname=graph_name,
        show=False)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_pq_based_ramp_exr()
    # plot_Ae_HDR10_out_with_white(
    #     graph_name="./img/Ae_Out_ProRes.png")
    # plot_Ae_HDR10_exr_out_with_white(
    #     graph_name="./img/Ae_Out_Exr_L.png")
    # plot_Ae_HDR10_exr_nl_out_with_white(
    #     graph_name="./img/Ae_Out_Exr_NL_png")
    plot_Ae_HDR10_exr_nl_out_vari_prj_white(
        graph_name="./img/Ae_working_space_tf.png")
