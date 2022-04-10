# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
from ctypes.wintypes import RGB
import os
import subprocess
from pathlib import Path


# import third-party libraries
import numpy as np
from colour.models import BT709_COLOURSPACE, RGB_COLOURSPACES
from colour.models import eotf_BT1886
from colour import XYZ_to_xyY, RGB_to_RGB
import cv2

# import my libraries
import plot_utility as pu
import test_pattern_generator2 as tpg
import transfer_functions as tf
import font_control as fc1
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


FFMPEG_NORMALIZE_COEF = 65340
const_lab_delta = 6.0/29.0
BIT_DEPTH = 8
CODE_VALUE_NUM = (2 ** BIT_DEPTH)
MAX_CODE_VALUE = CODE_VALUE_NUM - 1
COLOR_CHECKER_H_HUM = 6
RGBMYC_COLOR_LIST = np.array(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1],
     [1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=np.uint8)
COLOR_CHECKER_LINEAR = tpg.generate_color_checker_rgb_value(
    color_space=BT709_COLOURSPACE)


def create_debug_img_with_icc_profile():
    fname = "./img/src_bt709_gm24.png"
    fname_with_profile = "./img/src_bt709_gm24_with_profile.png"
    icc_profile = "./icc_profile/Gamma2.4_BT.709_D65.icc"
    cmd = [
        'convert', fname, '-define', "png:color-type=2",
        '-profile', icc_profile, fname_with_profile]
    subprocess.run(cmd)


def create_debug_img_with_icc_profile2():
    fname = "./img/src_bt2020_gm24.png"
    fname_with_profile = "./img/src_bt2020_gm24_with_profile.png"
    icc_profile = "./icc_profile/Rec2020-g24-v4.icc"
    cmd = [
        'convert', fname, '-define', "png:color-type=2",
        '-profile', icc_profile, fname_with_profile]
    subprocess.run(cmd)


def create_debug_img_with_icc_profile3():
    fname = "./img/src_bt709_gm24.png"
    fname_with_profile = "./img/src_bt709_gm24_with_profile_icc.png"
    icc_profile = "./icc_profile/ITU-RBT709ReferenceDisplay.icc"
    cmd = [
        'convert', fname, '-define', "png:color-type=2",
        '-profile', icc_profile, fname_with_profile]
    subprocess.run(cmd)


def create_debug_img_with_icc_profile4():
    fname = "./img/src_ap0_gm35.png"
    fname_with_profile = "./img/src_ap0_gm35_with_profile_icc.png"
    icc_profile = "./icc_profile/Gamma3.5_ACES-AP0_D65.icc"
    cmd = [
        'convert', fname, '-define', "png:color-type=2",
        '-profile', icc_profile, fname_with_profile]
    subprocess.run(cmd)


def create_debug_img_with_icc_profile5():
    fname = "./img/src_bt709_srgb.png"
    fname_with_profile = "./img/src_bt709_srgb_with_profile_icc.png"
    icc_profile = "./icc_profile/sRGB_ty.icc"
    cmd = [
        'convert', fname, '-define', "png:color-type=2",
        '-profile', icc_profile, fname_with_profile]
    subprocess.run(cmd)


def main_func():
    # create_youtube_srgb_gm24_pattern()
    # create_youtube_pattern_appearance(left_cv_in=12, right_cv_in=32)
    # create_youtube_pattern_appearance(left_cv_in=32, right_cv_in=48)
    # create_youtube_pattern_appearance(left_cv_in=0, right_cv_in=12)
    # create_youtube_pattern_appearance(left_cv_in=0, right_cv_in=32)
    # create_debug_img_with_icc_profile()
    # create_debug_img_with_icc_profile3()
    # create_debug_img_with_icc_profile4()
    # create_debug_img_with_icc_profile5()

    # analyze_browser_cms_result(
    #     fname="./src_img/photoshop_gm24_win11.png",
    #     env="Photoshop, Win 11",
    #     gamut_name=cs.BT709, gamma_name=tf.GAMMA24)

    # analyze_browser_cms_result(
    #     fname="./src_img/chrome_gm24_win11.png",
    #     env="Chrome, Win 11",
    #     gamut_name=cs.BT709, gamma_name=tf.GAMMA24)

    # analyze_browser_cms_result(
    #     fname="./src_img/firefox_gm24_win11.png",
    #     env="Firefox, Win 11",
    #     gamut_name=cs.BT709, gamma_name=tf.GAMMA24)

    # analyze_browser_cms_result(
    #     fname="./src_img/chrome_gm24_iOS.png",
    #     env="Chrome, iOS 15.4",
    #     gamut_name=cs.BT709, gamma_name=tf.GAMMA24)
    # analyze_browser_cms_result(
    #     fname="./src_img/Firefox_gm24_iOS.png",
    #     env="Firefox, iOS 15.4",
    #     gamut_name=cs.BT709, gamma_name=tf.GAMMA24)
    # analyze_browser_cms_result(
    #     fname="./src_img/chrome_gm24_Android.png",
    #     env="Chrome, Android 12",
    #     gamut_name=cs.BT709, gamma_name=tf.GAMMA24)
    # analyze_browser_cms_result(
    #     fname="./src_img/Firefox_gm24_Android.png",
    #     env="Firefox, Android 12",
    #     gamut_name=cs.BT709, gamma_name=tf.GAMMA24)
    # analyze_browser_cms_result(
    #     fname="./src_img/Safari_gm24_iOS.png",
    #     env="Safari, iOS 15.4",
    #     gamut_name=cs.BT709, gamma_name=tf.GAMMA24)

    # analyze_browser_cms_result(
    #     fname="./src_img/Chrome_GM24_Win10.png",
    #     env="Chrome, Win 10",
    #     gamut_name=cs.BT709, gamma_name=tf.GAMMA24)
    # analyze_browser_cms_result(
    #     fname="./src_img/Firefox_GM24_Win10.png",
    #     env="Firefox, Win 10",
    #     gamut_name=cs.BT709, gamma_name=tf.GAMMA24)

    # analyze_browser_cms_result(
    #     fname="./src_img/affinity_gm24_win11.png",
    #     env="Affinity Photo, Win 11",
    #     gamut_name=cs.BT709, gamma_name=tf.GAMMA24)

    # analyze_browser_cms_result(
    #     fname="./src_img/chrome_gm24_icc_ref_win11.png",
    #     env="Chrome_ICC_Ref_Profile, Win 11",
    #     gamut_name=cs.BT709, gamma_name=tf.GAMMA24)

    # analyze_browser_cms_result(
    #     fname="./src_img/chrome_ap0_gm35_win11.png",
    #     env="Chrome, Win 11",
    #     gamut_name=cs.ACES_AP0, gamma_name=tf.GAMMA35)
    # analyze_browser_cms_result(
    #     fname="./src_img/firefox_ap0_gm35_win11.png",
    #     env="Firefox, Win 11",
    #     gamut_name=cs.ACES_AP0, gamma_name=tf.GAMMA35)
    # analyze_browser_cms_result(
    #     fname="./src_img/photosop_ap0_gm35_win11.png",
    #     env="Photoshop, Win 11",
    #     gamut_name=cs.ACES_AP0, gamma_name=tf.GAMMA35)

    # analyze_browser_cms_result(
    #     fname="./src_img/chrome_srgb_win11.png",
    #     env="Chrome, Win 11",
    #     gamut_name=cs.BT709, gamma_name=tf.SRGB)
    # analyze_browser_cms_result(
    #     fname="./src_img/firefox_srgb_win11.png",
    #     env="Firefox, Win 11",
    #     gamut_name=cs.BT709, gamma_name=tf.SRGB)
    # analyze_browser_cms_result(
    #     fname="./src_img/photosop_srgb_win11.png",
    #     env="Photoshop, Win 11",
    #     gamut_name=cs.BT709, gamma_name=tf.SRGB)

    # analyze_browser_cms_result_with_in_out(
    #     fname="./src_img/photoshop_srgb_to_ap0-gm35_win11.png",
    #     env="Photoshop, Win 11",
    #     in_gamut_name=cs.BT709, out_gamut_name=cs.ACES_AP0,
    #     in_gamma_name=tf.SRGB, out_gamma_name=tf.GAMMA35)
    # analyze_browser_cms_result_with_in_out(
    #     fname="./src_img/chrome_srgb_to_ap0-gm35_win11.png",
    #     env="Chrome, Win 11",
    #     in_gamut_name=cs.BT709, out_gamut_name=cs.ACES_AP0,
    #     in_gamma_name=tf.SRGB, out_gamma_name=tf.GAMMA35)
    # analyze_browser_cms_result_with_in_out(
    #     fname="./src_img/firefox_srgb_to_ap0-gm35_win11.png",
    #     env="Firefox, Win 11",
    #     in_gamut_name=cs.BT709, out_gamut_name=cs.ACES_AP0,
    #     in_gamma_name=tf.SRGB, out_gamma_name=tf.GAMMA35)

    # analyze_browser_cms_result_with_in_out(
    #     fname="./src_img/photoshop_ap0-gm35_to_srgb_win11.png",
    #     env="Photoshop, Win 11",
    #     in_gamut_name=cs.ACES_AP0, out_gamut_name=cs.BT709,
    #     in_gamma_name=tf.GAMMA35, out_gamma_name=tf.SRGB)
    # analyze_browser_cms_result_with_in_out(
    #     fname="./src_img/chrome_ap0-gm35_to_srgb_win11.png",
    #     env="Chrome, Win 11",
    #     in_gamut_name=cs.ACES_AP0, out_gamut_name=cs.BT709,
    #     in_gamma_name=tf.GAMMA35, out_gamma_name=tf.SRGB)
    # analyze_browser_cms_result_with_in_out(
    #     fname="./src_img/firefox_ap0-gm35_to_srgb_win11.png",
    #     env="Firefox, Win 11",
    #     in_gamut_name=cs.ACES_AP0, out_gamut_name=cs.BT709,
    #     in_gamma_name=tf.GAMMA35, out_gamma_name=tf.SRGB)

    # analyze_browser_cms_result_with_in_out(
    #     fname="./src_img/photoshop_ap0_gm35_win11.png",
    #     env="Photoshop v23.2, Win 11",
    #     in_gamut_name=cs.ACES_AP0, out_gamut_name=cs.ACES_AP0,
    #     in_gamma_name=tf.GAMMA35, out_gamma_name=tf.GAMMA35)
    # analyze_browser_cms_result_with_in_out(
    #     fname="./src_img/photoshop_ap0_gm35_to_srgb_win11.png",
    #     env="Photoshop v23.2, Win 11",
    #     in_gamut_name=cs.ACES_AP0, out_gamut_name=cs.BT709,
    #     in_gamma_name=tf.GAMMA35, out_gamma_name=tf.SRGB)

    # analyze_browser_cms_result_with_in_out(
    #     fname="./src_img/photoshop_srgb_srgb_to_ap0_srgb_win11.png",
    #     env="Photoshop, Win 11",
    #     in_gamut_name=cs.BT709, out_gamut_name=cs.ACES_AP0,
    #     in_gamma_name=tf.SRGB, out_gamma_name=tf.SRGB)
    # analyze_browser_cms_result_with_in_out(
    #     fname="./src_img/chrome_srgb_srgb_to_ap0_srgb_win11.png",
    #     env="Chrome, Win 11",
    #     in_gamut_name=cs.BT709, out_gamut_name=cs.ACES_AP0,
    #     in_gamma_name=tf.SRGB, out_gamma_name=tf.SRGB)
    # analyze_browser_cms_result_with_in_out(
    #     fname="./src_img/firefox_srgb_srgb_to_ap0_srgb_win11.png",
    #     env="Firefox, Win 11",
    #     in_gamut_name=cs.BT709, out_gamut_name=cs.ACES_AP0,
    #     in_gamma_name=tf.SRGB, out_gamma_name=tf.SRGB)

    # analyze_browser_cms_result_with_in_out(
    #     fname="./src_img/photoshop_srgb_2.4_to_ap0_srgb_win11.png",
    #     env="Photoshop, Win 11",
    #     in_gamut_name=cs.BT709, out_gamut_name=cs.ACES_AP0,
    #     in_gamma_name=tf.GAMMA24, out_gamma_name=tf.SRGB)
    # analyze_browser_cms_result_with_in_out(
    #     fname="./src_img/chrome_srgb_2.4_to_ap0_srgb_win11.png",
    #     env="Chrome, Win 11",
    #     in_gamut_name=cs.BT709, out_gamut_name=cs.ACES_AP0,
    #     in_gamma_name=tf.GAMMA24, out_gamma_name=tf.SRGB)
    # analyze_browser_cms_result_with_in_out(
    #     fname="./src_img/chrome_bt709_srgb_to_bt709_srgb.png",
    #     env="Chrome, Win 11",
    #     in_gamut_name=cs.BT709, out_gamut_name=cs.BT709,
    #     in_gamma_name=tf.SRGB, out_gamma_name=tf.SRGB)
    # analyze_browser_cms_result_with_in_out(
    #     fname="./src_img/chrome_bt709_2.4_to_bt709_srgb.png",
    #     env="Chrome, Win 11",
    #     in_gamut_name=cs.BT709, out_gamut_name=cs.BT709,
    #     in_gamma_name=tf.GAMMA24, out_gamma_name=tf.SRGB)
    # analyze_browser_cms_result_with_in_out(
    #     fname="./src_img/chrome_ap0_3.5_to_bt709_srgb.png",
    #     env="Chrome, Win 11",
    #     in_gamut_name=cs.ACES_AP0, out_gamut_name=cs.BT709,
    #     in_gamma_name=tf.GAMMA35, out_gamma_name=tf.SRGB)

    # analyze_browser_cms_result_with_in_out(
    #     fname="./src_img/chrome_bt709_srgb_to_bt709_srgb.png",
    #     env="Chrome, Win 11",
    #     in_gamut_name=cs.BT709, out_gamut_name=cs.BT709,
    #     in_gamma_name=tf.SRGB, out_gamma_name=tf.SRGB)
    # analyze_browser_cms_result_with_in_out(
    #     fname="./src_img/chrome_bt709_srgb_to_bt709_2.4.png",
    #     env="Chrome, Win 11",
    #     in_gamut_name=cs.BT709, out_gamut_name=cs.BT709,
    #     in_gamma_name=tf.SRGB, out_gamma_name=tf.GAMMA24)
    # analyze_browser_cms_result_with_in_out(
    #     fname="./src_img/chrome_bt709_srgb_to_ap0_3.5.png",
    #     env="Chrome, Win 11",
    #     in_gamut_name=cs.BT709, out_gamut_name=cs.ACES_AP0,
    #     in_gamma_name=tf.SRGB, out_gamma_name=tf.GAMMA35)

    # analyze_browser_cms_result_with_in_out(
    #     fname="./src_img/chrome_bt709_srgb_to_ap0_srgb.png",
    #     env="Chrome, Win 11",
    #     in_gamut_name=cs.BT709, out_gamut_name=cs.ACES_AP0,
    #     in_gamma_name=tf.SRGB, out_gamma_name=tf.SRGB)

    # analyze_browser_cms_result_with_in_out(
    #     fname="./src_img/photoshop_ap0_3.5_to_bt709_srgb.png",
    #     env="Photoshop v23.2, Win 11",
    #     in_gamut_name=cs.ACES_AP0, out_gamut_name=cs.BT709,
    #     in_gamma_name=tf.GAMMA35, out_gamma_name=tf.SRGB)

    # analyze_browser_cms_result_with_in_out(
    #     fname="./src_img/affinity_ap0_3.5_to_bt709_srgb.png",
    #     env="Affinity Photo v1.10.5, Win 11",
    #     in_gamut_name=cs.ACES_AP0, out_gamut_name=cs.BT709,
    #     in_gamma_name=tf.GAMMA35, out_gamma_name=tf.SRGB)

    # analyze_browser_cms_result_with_in_out(
    #     fname="./src_img/chrome_bt709_srgb_to_p3d65_srgb.png",
    #     env="Chrome, Win 11",
    #     in_gamut_name=cs.BT709, out_gamut_name=cs.P3_D65,
    #     in_gamma_name=tf.SRGB, out_gamma_name=tf.SRGB)

    analyze_browser_cms_result_with_in_out(
        fname="./src_img/chrome_bt709_2.4_to_bt709_2.4.png",
        env="Chrome, Win 11",
        in_gamut_name=cs.BT709, out_gamut_name=cs.BT709,
        in_gamma_name=tf.GAMMA24, out_gamma_name=tf.GAMMA24)


def debug_func():
    # plot_srgb_gm24_oetf()
    # plot_srgb_gm24_oetf_all()
    # plot_srgb_to_gm24()
    # plot_gm35_to_sRGB()
    # plot_bt1886_plus_alpha()
    # create_0cv_to_8cv()
    plot_average()


def plot_average():
    average_list = np.zeros(9)
    for idx in range(9):
        average_list[idx] = crop_and_average_benq_each_file(idx)
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=None,
        graph_title_size=None,
        xlabel="Code Value (8 bit)",
        ylabel="Average (Captured code value)",
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
    ax1.plot(average_list, '-o')
    pu.show_and_save(
        fig=fig, legend_loc=None, save_fname="./graph/average_plot.png")


def crop_and_average_benq_each_file(idx=0):
    crop_pos_h = 2468 + 250
    crop_pos_v = 1438 + 250
    crop_len = 200

    fname = f"./increment_cv_img/result/{idx:02}CV.JPG"
    base_img = tpg.img_read(fname)
    img = base_img[
        crop_pos_v:crop_pos_v+crop_len,
        crop_pos_h:crop_pos_h+crop_len,
        1].flatten()
    # print(f"average = {np.average(img)}")

    return np.average(img)

    # fig, ax1 = pu.plot_1_graph(
    #     fontsize=20,
    #     figsize=(10, 8),
    #     bg_color=(0.96, 0.96, 0.96),
    #     graph_title=f"Shooting result ({idx:02} CV)",
    #     graph_title_size=None,
    #     xlabel="Code Value (8 bit)", ylabel="Frequency",
    #     axis_label_size=None,
    #     legend_size=17,
    #     xlim=None,
    #     ylim=(0, 8500),
    #     xtick=None,
    #     ytick=None,
    #     xtick_size=None, ytick_size=None,
    #     linewidth=3,
    #     minor_xtick_num=None,
    #     minor_ytick_num=None)
    # ax1.hist(img, range=(40, 120), bins=80)
    # pu.show_and_save(
    #     fig=fig, legend_loc='upper left',
    #     save_fname=f"./graph/hist_{idx:02d}.png")


def create_0cv_to_8cv():
    width = 1920
    height = 1080
    max_cv = 8
    img_base = np.ones((height, width, 3), dtype=np.uint8)
    for idx in range(max_cv + 1):
        img = img_base * idx
        fname = f"./increment_cv_img/img_{idx:04d}_cv.png"
        print(fname)
        tpg.img_write(fname, img)


def calc_gm24_with_black(x, lw=100, lb=0.1):
    gm24_ref = tf.eotf(x, tf.GAMMA24)
    gm24 = gm24_ref * (lw - lb) + lb

    return gm24


def plot_bt1886_plus_alpha():
    lw = 100
    x = np.linspace(0, 1, 256)
    x_8bit = x * 255
    gm24_ref = tf.eotf_to_luminance(x, tf.GAMMA24)
    # gm24_lb_01 = calc_gm24_with_black(x, lw=lw, lb=0.1)

    # bt86_lb0 = tf.bt1886_eotf(x, lw, 0.0)
    # bt86_lb01 = tf.bt1886_eotf(x, lw, 0.1)
    bt86_000 = eotf_BT1886(x, L_B=0.00, L_W=lw)
    bt86_001 = eotf_BT1886(x, L_B=0.01, L_W=lw)
    bt86_010 = eotf_BT1886(x, L_B=0.10, L_W=lw)

    x_max = 256
    x_min = -x_max * 0.04
    x_tick_unit = x_max // 8
    x_tick_num = (x_max // x_tick_unit) + 1
    y_max = 105
    y_min = -y_max * 0.04

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="BT.1886",
        graph_title_size=None,
        xlabel="Code Value (8 bit)",
        ylabel="Luminance [cd/m2]",
        axis_label_size=None,
        legend_size=17,
        xlim=[x_min, x_max],
        ylim=[y_min, y_max],
        xtick=[x * x_tick_unit for x in range(x_tick_num)],
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    # ax1.plot(x_8bit, gm24_ref, '-', color=pu.RED, label="Lb=0")
    # ax1.plot(x_8bit, gm24_lb_01, '-', color=pu.GREEN, label="Lb=0.1")
    # ax1.plot(x_8bit, bt86_lb0, '-', color=pu.SKY, label="BT.1886 Lb=0.0")
    ax1.plot(
        x_8bit, bt86_000, '-', color=pu.RED,
        label="BT.1886 L_B=0.0, L_W=100")
    # ax1.plot(
    #     x_8bit, bt86_001, '-', color=pu.GREEN,
    #     label="BT.1886 L_B=0.01, L_W=100")
    ax1.plot(
        x_8bit, bt86_010, '-', color=pu.SKY,
        label="BT.1886 L_B=0.1, L_W=100")
    pu.show_and_save(
        fig=fig, legend_loc='upper left',
        save_fname="./img/bt1886_plus.png")


def plot_srgb_to_gm24():
    x = np.linspace(0, 1, 256)
    y = tf.eotf_to_luminance(x, tf.GAMMA24)
    x2 = tf.oetf_from_luminance(y, tf.SRGB)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(9, 9),
        bg_color=(0.7, 0.7, 0.7),
        graph_title="Gamma 2.4 to sRGB",
        graph_title_size=None,
        xlabel="Input Code Value (8bit)",
        ylabel="Output Code Value (8bit)",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=[x * 32 for x in range(256//32 + 1)],
        ytick=[x * 32 for x in range(256//32 + 1)],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.xaxis.grid(which='major', color=(0.4, 0.4, 0.4))
    ax1.yaxis.grid(which='major', color=(0.4, 0.4, 0.4))
    ax1.plot(
        x * 255, x * 255, '--', color=pu.YELLOW,
        label="Reference Value", lw=2, zorder=10)
    ax1.plot(
        x * 255, x2 * 255, '-o', color=pu.BLUE,
        label="Gamma 2.4 to sRGB", lw=2)
    pu.show_and_save(
        fig=fig, legend_loc=None, save_fname="./graph/GM24_to_sRGB.png")


def plot_gm35_to_sRGB():
    x = np.linspace(0, 1, 256)
    y = tf.eotf_to_luminance(x, tf.GAMMA35)
    x2 = tf.oetf_from_luminance(y, tf.SRGB)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(9, 9),
        bg_color=(0.7, 0.7, 0.7),
        graph_title="Gamma 3.5 to sRGB",
        graph_title_size=None,
        xlabel="Input Code Value (8bit)",
        ylabel="Output Code Value (8bit)",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=[x * 32 for x in range(256//32 + 1)],
        ytick=[x * 32 for x in range(256//32 + 1)],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.xaxis.grid(which='major', color=(0.4, 0.4, 0.4))
    ax1.yaxis.grid(which='major', color=(0.4, 0.4, 0.4))
    ax1.plot(
        x * 255, x * 255, '--', color=pu.YELLOW,
        label="No Conversion", lw=2, zorder=10)
    ax1.plot(
        x * 255, x2 * 255, '-o', color=pu.BLUE,
        label="Gamma 3.5 to sRGB", lw=2)
    pu.show_and_save(
        fig=fig, legend_loc=None, save_fname="./graph/GM35_to_sRGB.png")


def plot_srgb_gm24_oetf():
    peak_lumiannce = 100
    contrast_ratio = 1/1000
    black_lumiannce = peak_lumiannce * contrast_ratio
    valid_lumi_range_ratio = (peak_lumiannce - black_lumiannce)/peak_lumiannce
    x = np.linspace(0, 1, 256)
    x_8bit = x * 255
    srgb_abs = tf.eotf_to_luminance(x, tf.SRGB)
    srgb = srgb_abs * valid_lumi_range_ratio + black_lumiannce
    gm24_abs = tf.eotf_to_luminance(x, tf.GAMMA24)
    gm24 = gm24_abs * valid_lumi_range_ratio + black_lumiannce
    pq = tf.eotf_to_luminance(x, tf.ST2084)
    x_max = 20
    x_min = -x_max * 0.04
    x_tick_unit = 2
    x_tick_num = (x_max // x_tick_unit) + 1
    y_max = 0.5
    y_min = -y_max * 0.04

    h_line_list = []
    for idx in range(x_max + 32):
        y_val = pq[idx]
        points = [[x_min, x_max], [y_val, y_val]]
        h_line_list.append(points)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=None,
        graph_title_size=None,
        xlabel="Code Value (8 bit)",
        ylabel="Luminance [cd/m2]",
        axis_label_size=None,
        legend_size=17,
        xlim=[x_min, x_max],
        ylim=[y_min, y_max],
        xtick=[x * x_tick_unit for x in range(x_tick_num)],
        ytick=[0, 0.1, 0.5],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.yaxis.grid(None)
    ax1.plot(
        x_8bit, srgb, '-o', color=pu.RED,
        label="sRGB (black: 0.1 cd/m2, white: 100 cd/m2)")
    ax1.plot(
        x_8bit, gm24, '-o', color=pu.SKY,
        label="Gamma 2.4 (black: 0.1 cd/m2, white: 100 cd/m2)")
    # ax1.plot(x_8bit, lstar, '-o', color=pu.BROWN, label="CIE 1976 L*")
    ax1.plot(
        x_8bit, pq, '-o', color=pu.GREEN,
        label="SMPTE ST 2084 (black: 0.0 cd/m2, white: 10,000 cd/m2)")
    for points in h_line_list:
        ax1.plot(points[0], points[1], '--', lw=0.5, color='k')
    pu.show_and_save(
        fig=fig, legend_loc='upper left',
        save_fname="./img/srgb_vs_gm24.png")


def plot_srgb_gm24_oetf_all():
    peak_lumiannce = 100
    contrast_ratio = 1/1000
    black_lumiannce = peak_lumiannce * contrast_ratio
    valid_lumi_range_ratio = (peak_lumiannce - black_lumiannce)/peak_lumiannce
    x = np.linspace(0, 1, 256)
    x_8bit = x * 255
    srgb_abs = tf.eotf_to_luminance(x, tf.SRGB)
    srgb = srgb_abs * valid_lumi_range_ratio + black_lumiannce
    gm24_abs = tf.eotf_to_luminance(x, tf.GAMMA24)
    gm24 = gm24_abs * valid_lumi_range_ratio + black_lumiannce
    pq = tf.eotf_to_luminance(x, tf.ST2084)
    x_max = 256
    x_min = -x_max * 0.04
    x_tick_unit = 32
    x_tick_num = (x_max // x_tick_unit) + 1
    y_max = 105
    y_min = -y_max * 0.04

    h_line_list = []
    for idx in range(x_max):
        y_val = pq[idx]
        points = [[x_min, x_max], [y_val, y_val]]
        h_line_list.append(points)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=None,
        graph_title_size=None,
        xlabel="Code Value (8 bit)",
        ylabel="Luminance [cd/m2]",
        axis_label_size=None,
        legend_size=17,
        xlim=[x_min, x_max],
        ylim=[y_min, y_max],
        xtick=[x * x_tick_unit for x in range(x_tick_num)],
        ytick=[x * 10 for x in range(11)],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    # ax1.yaxis.grid(None)
    ax1.plot(
        x_8bit, srgb, '-o', color=pu.RED,
        label="sRGB (black: 0.1 cd/m2, white: 100 cd/m2)")
    ax1.plot(
        x_8bit, gm24, '-o', color=pu.SKY,
        label="Gamma 2.4 (black: 0.1 cd/m2, white: 100 cd/m2)")
    # ax1.plot(x_8bit, lstar, '-o', color=pu.BROWN, label="CIE 1976 L*")
    ax1.plot(
        x_8bit, pq, '-o', color=pu.GREEN,
        label="SMPTE ST 2084 (black: 0.0 cd/m2, white: 10,000 cd/m2)")
    # for points in h_line_list:
    #     ax1.plot(points[0], points[1], '--', lw=0.5, color='k')
    pu.show_and_save(
        fig=fig, legend_loc='upper left',
        save_fname="./img/srgb_vs_gm24_all.png")


def create_tp_main(
        width=960, height=1080, cv1=10, cv2=20, tp_size_ratio=0.5,
        tile_num=6, text="sample", bg_cv=32):
    """
    non-linear space
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * bg_cv
    tp_size = int(width * tp_size_ratio + 0.5)
    tp_img = tpg.make_tile_pattern(
        width=tp_size, height=tp_size, low_level=cv1, high_level=cv2,
        h_tile_num=tile_num, v_tile_num=tile_num, dtype=np.uint8)
    # tpg.draw_outline(tp_img, fg_color=[32, 32, 32], outline_width=1)

    merge_pos = [(width//2) - (tp_size//2), (height//2) - (tp_size//2)]
    tpg.merge(img, tp_img, merge_pos)

    # add text
    font = fc1.NOTO_SANS_MONO_BOLD
    font_size = 40
    font_edge_size = 6
    font_edge_color = 'black'
    img = img / 255
    _, text_height = fc1.get_text_size(
        text=text, font_size=font_size, font_path=font,
        stroke_width=font_edge_size, stroke_fill=font_edge_color)
    text_pos = [
        merge_pos[0], merge_pos[1] - int(text_height * 1.1)]
    text_drawer = fc1.TextDrawer(
        img=img, text=text, pos=text_pos,
        font_color=(64/255, 64/255, 64/255), font_size=font_size,
        font_path=font,
        stroke_width=font_edge_size, stroke_fill=font_edge_color)
    text_drawer.draw()
    img = np.uint8(np.round(img * 255))

    return img


def create_youtube_srgb_gm24_pattern():
    width = 1920
    height = 1080
    half_width = width // 2
    left_cv = [0, 6]
    right_cv_base = np.array([0, 6])
    right_cv = np.uint8(
        np.round(
            tf.oetf_from_luminance(
                tf.eotf_to_luminance(right_cv_base/255, tf.SRGB),
                tf.GAMMA24)
            * 255))
    print(f"lr = {left_cv, right_cv}")
    tp_size_ratio = 0.7
    tile_num = 8

    left_text = f"{left_cv[0]} CV and {left_cv[1]} CV (Gamma 2.4)"
    right_text = f"{right_cv_base[0]} CV and {right_cv_base[1]} CV (sRGB)"
    left_img = create_tp_main(
        width=half_width, height=height,
        cv1=left_cv[0], cv2=left_cv[1], tp_size_ratio=tp_size_ratio,
        tile_num=tile_num, text=left_text)
    right_img = create_tp_main(
        width=half_width, height=height,
        cv1=right_cv[0], cv2=right_cv[1], tp_size_ratio=tp_size_ratio,
        tile_num=tile_num, text=right_text)

    img = np.hstack((left_img, right_img))

    fname = "./img/tp_sample.png"
    icc_profile = "./icc_profile/Gamma2.4_BT.709_D65.icc"
    fname_with_profile = "./img/YouTube_GM24_sRGB_Descrimination_tp.png"
    tpg.img_write(fname, img)
    cmd = [
        'convert', fname, '-define', "png:color-type=2",
        '-profile', icc_profile, fname_with_profile]
    subprocess.run(cmd)


def create_youtube_pattern_appearance(left_cv_in=24, right_cv_in=48):
    width = 1920
    height = 1080
    half_width = width // 2
    left_cv = [0, left_cv_in]
    right_cv = [0, right_cv_in]
    print(f"lr = {left_cv, right_cv}")
    tp_size_ratio = 0.7
    tile_num = 8
    bg_cv = 40

    left_text = f"{left_cv[0]} CV and {6} CV (Gamma 2.4)"
    right_text = f"{0} CV and {6} CV (sRGB)"
    left_img = create_tp_main(
        width=half_width, height=height,
        cv1=left_cv[0], cv2=left_cv[1], tp_size_ratio=tp_size_ratio,
        tile_num=tile_num, text=left_text, bg_cv=bg_cv)
    right_img = create_tp_main(
        width=half_width, height=height,
        cv1=right_cv[0], cv2=right_cv[1], tp_size_ratio=tp_size_ratio,
        tile_num=tile_num, text=right_text, bg_cv=bg_cv)

    img = np.hstack((left_img, right_img))

    fname = "./img/appearance_base.png"
    icc_profile = "./icc_profile/sRGB_ty.icc"
    fname_with_profile\
        = f"./img/YouTube_Pattern_Appearance_{left_cv_in}-{right_cv_in}.png"
    tpg.img_write(fname, img)
    cmd = [
        'convert', fname, '-define', "png:color-type=2",
        '-profile', icc_profile, fname_with_profile]
    subprocess.run(cmd)


def create_alpha_black(alpha=128):
    width = 940
    height = 1080
    img = np.ones((height, width, 4), dtype=np.uint8)
    img[:, :, :3] = 0
    img[:, :, 3] = alpha

    fname = f"./img/bg_alpha-{alpha}.png"
    tpg.img_write(fname, img)


def calc_block_num_h(width=1920, block_size=64):
    return width // block_size


def calc_gradation_pattern_block_st_pos(
        code_value=MAX_CODE_VALUE, width=1920, height=1080, block_size=64):
    block_num_h = calc_block_num_h(width=width, block_size=block_size)
    st_pos_h = (code_value % block_num_h) * block_size
    st_pos_v = (code_value // block_num_h) * block_size
    st_pos = (st_pos_h, st_pos_v)

    return st_pos


def calc_rgbmyc_pattern_block_st_pos(
        color_idx=1, width=1920, height=1080, block_size=64):
    block_num_h = calc_block_num_h(width=width, block_size=block_size)
    st_pos_v = ((MAX_CODE_VALUE // block_num_h) + 2) * block_size
    st_pos_h = (color_idx % block_num_h) * block_size
    st_pos = (st_pos_h, st_pos_v)

    return st_pos


def calc_color_checker_pattern_block_st_pos(
        color_idx=1, width=1920, height=1080, block_size=64):
    block_num_h = calc_block_num_h(width=width, block_size=block_size)
    st_pos_v_offset = ((MAX_CODE_VALUE // block_num_h) + 4) * block_size
    st_pos_v = (color_idx // COLOR_CHECKER_H_HUM) * block_size\
        + st_pos_v_offset
    st_pos_h = (color_idx % COLOR_CHECKER_H_HUM) * block_size
    st_pos = (st_pos_h, st_pos_v)

    return st_pos


def get_specific_pos_value(img, pos):
    """
    Parameters
    ----------
    img : ndarray
        image data.
    pos : list
        pos[0] is horizontal coordinate, pos[1] is verical coordinate.
    """
    return img[pos[1], pos[0]]


def read_code_value_from_gradation_pattern(fname):
    """
    Example
    -------
    >>> read_code_value_from_gradation_pattern(
    ...     fname="./data.png, width=1920, height=1080, block_size=64)
    {'ramp': array(
          [[[  0,   0,   0],
            [  1,   1,   1],
            [  2,   2,   2],
            [  3,   3,   3],
            ...
            [252, 252, 252],
            [253, 253, 253],
            [254, 254, 254],
            [255, 255, 255]]], dtype=uint8),
     'rgbmyc': array(
           [[255,   0,   0],
            [  0, 255,   0],
            [  0,   0, 255],
            [255,   0, 255],
            [255, 255,   0],
            [  0, 255, 255]], dtype=uint8),
     'colorchecker': array(
           [[123,  90,  77],
            [201, 153, 136],
            [ 99, 129, 161],
            [ 98, 115,  75],
            ...
            [166, 168, 168],
            [128, 128, 129],
            [ 91,  93,  95],
            [ 59,  60,  61]], dtype=uint8)}
    """
    # define
    width = 1920
    height = 1080
    block_size = 64

    # Gradation
    print(f"reading {fname}")
    img = tpg.img_read(fname)
    img = cv2.resize(
        img, (width, height), interpolation=cv2.INTER_NEAREST)

    block_offset = block_size // 2
    ramp_value = np.zeros((1, CODE_VALUE_NUM, 3), dtype=np.uint8)
    for code_value in range(CODE_VALUE_NUM):
        st_pos = calc_gradation_pattern_block_st_pos(
            code_value=code_value, width=width, height=height,
            block_size=block_size)
        center_pos = (st_pos[0] + block_offset, st_pos[1] + block_offset)
        ramp_value[0, code_value] = get_specific_pos_value(img, center_pos)

    # RGBMYC
    rgbmyc_value = np.zeros_like(RGBMYC_COLOR_LIST)
    for color_idx in range(len(RGBMYC_COLOR_LIST)):
        st_pos = calc_rgbmyc_pattern_block_st_pos(
            color_idx=color_idx, width=width, height=height,
            block_size=block_size)
        center_pos = (st_pos[0] + block_offset, st_pos[1] + block_offset)
        rgbmyc_value[color_idx] = get_specific_pos_value(img, center_pos)
    rgbmyc_value = rgbmyc_value.reshape(
        1, rgbmyc_value.shape[0], rgbmyc_value.shape[1])

    # ColorChecker
    color_checker_value = np.zeros_like(COLOR_CHECKER_LINEAR, dtype=np.uint8)
    for color_idx in range(len(COLOR_CHECKER_LINEAR)):
        st_pos = calc_color_checker_pattern_block_st_pos(
            color_idx=color_idx, width=width, height=height,
            block_size=block_size)
        center_pos = (st_pos[0] + block_offset, st_pos[1] + block_offset)
        color_checker_value[color_idx] = get_specific_pos_value(
            img, center_pos)
    color_checker_value = color_checker_value.reshape(
        1, color_checker_value.shape[0], color_checker_value.shape[1])

    return dict(
        ramp=ramp_value, rgbmyc=rgbmyc_value,
        colorchecker=color_checker_value)


def create_result_graph_name(in_name, suffix='_gamma', output_dir="./graph"):
    """
    Example
    -------
    >>> create_result_graph_name(
    ...     "./img/src_bt709_gm24.png", suffix='_tf', output_dir="./graph")
    graph/src_bt709_gm24_tf.png
    """
    p_file = Path(in_name)
    ext = p_file.suffix
    basename = p_file.stem

    p_out_name = Path(output_dir) / (basename + suffix + ext)

    return str(p_out_name)


def plot_transfer_characteristics_cms_result(data, graph_name):
    x = np.arange(len(data[0]))
    y = x.copy()
    r = data[..., 0].flatten()
    g = data[..., 1].flatten()
    b = data[..., 2].flatten()
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(9, 9),
        bg_color=(0.7, 0.7, 0.7),
        graph_title="Input-Output Characteristics (Ramp)",
        graph_title_size=None,
        xlabel="Input Code Value (8 bit)",
        ylabel="Output Code Value (8 bit)",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=[x * 32 for x in range(256//32 + 1)],
        ytick=[x * 32 for x in range(256//32 + 1)],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.xaxis.grid(which='major', color=(0.4, 0.4, 0.4))
    ax1.yaxis.grid(which='major', color=(0.4, 0.4, 0.4))
    ax1.plot(x, y, '--', color=pu.YELLOW, lw=2, label="Reference", zorder=10)
    ax1.plot(x, r, '-o', color=pu.RED, label='Measured - Red')
    ax1.plot(x, g, '-o', color=pu.GREEN, label='Measured - Green')
    ax1.plot(x, b, '-o', color=pu.BLUE, label='Measured - Blue')
    pu.show_and_save(fig=fig, legend_loc='upper left', save_fname=graph_name)


def plot_transfer_characteristics_cms_result_with_in_out(
        data, in_tf, out_tf, graph_name):
    x = np.arange(len(data[0]))
    linear = tf.eotf(x/255, in_tf)
    y = np.uint8(np.round(tf.oetf(linear, out_tf) * 255))
    r = data[..., 0].flatten()
    g = data[..., 1].flatten()
    b = data[..., 2].flatten()
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(9, 9),
        bg_color=(0.7, 0.7, 0.7),
        graph_title="Input-Output Characteristics (Ramp)",
        graph_title_size=None,
        xlabel=f"Input Code Value ({in_tf})",
        ylabel=f"Output Code Value ({out_tf})",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=[x * 32 for x in range(256//32 + 1)],
        ytick=[x * 32 for x in range(256//32 + 1)],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.xaxis.grid(which='major', color=(0.4, 0.4, 0.4))
    ax1.yaxis.grid(which='major', color=(0.4, 0.4, 0.4))
    ax1.plot(x, y, '--', color=pu.YELLOW, lw=2, label="Reference", zorder=10)
    ax1.plot(x, r, '-o', color=pu.RED, label='Measured - Red')
    ax1.plot(x, g, '-o', color=pu.GREEN, label='Measured - Green')
    ax1.plot(x, b, '-o', color=pu.BLUE, label='Measured - Blue')

    # x2 = np.linspace(0, 1, 256)
    # y2 = tf.oetf((x2 ** 2.4), tf.SRGB) * 0xFF
    # ax1.plot(
    #     x, y2, '--', color=pu.SKY, lw=2, label='Gamma 2.4 to sRGB', zorder=11)
    pu.show_and_save(fig=fig, legend_loc='upper left', save_fname=graph_name)


def plot_gamut_cms_result(data_dict, gamut_name, gamma_name, graph_name):
    rgbmyc_num = 6
    color_checker_num = 19
    rate = 1.1

    data = np.hstack(
        [data_dict['rgbmyc'],
         data_dict['colorchecker'][:, :color_checker_num]])
    linear_rgb = tf.eotf(data / 255, gamma_name)
    marker_color = RGB_to_RGB(
        linear_rgb, RGB_COLOURSPACES[gamut_name], RGB_COLOURSPACES[cs.BT709])
    marker_color = tf.oetf(np.clip(marker_color, 0.0, 1.0), tf.GAMMA24)
    large_xyz = cs.calc_XYZ_from_rgb(
        rgb=linear_rgb, color_space_name=gamut_name)
    xyY = XYZ_to_xyY(large_xyz)

    rgbmyc_ref_rgb = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1]])
    color_checker_ref_rgb = tpg.generate_color_checker_rgb_value(
        color_space=RGB_COLOURSPACES[gamut_name], target_white=cs.D65)
    ref_rgb = np.hstack(
        [rgbmyc_ref_rgb.reshape(1, rgbmyc_num, 3),
         color_checker_ref_rgb[:color_checker_num].reshape(
             1, color_checker_num, 3)])
    ref_xyz = cs.calc_XYZ_from_rgb(
        rgb=ref_rgb, color_space_name=gamut_name)
    ref_xyY = XYZ_to_xyY(ref_xyz)

    # diagram
    xmin = 0.0
    xmax = 0.8
    ymin = 0.0
    ymax = 0.9
    cmf_xy = tpg._get_cmfs_xy()
    xlim = (min(0, xmin), max(0.8, xmax))
    ylim = (min(0, ymin), max(0.9, ymax))
    figsize_h = 8 * rate
    figsize_v = 9 * rate
    # gamut の用意
    outer_gamut, _ = tpg.get_primaries(cs.BT709)
    fig, ax1 = pu.plot_1_graph(
        bg_color=(0.8, 0.8, 0.8),
        fontsize=20 * rate,
        figsize=(figsize_h, figsize_v),
        graph_title="Color Checker and RGBMYC",
        xlabel=None, ylabel=None,
        legend_size=18 * rate,
        xlim=xlim, ylim=ylim,
        xtick=[x * 0.1 + xmin for x in
               range(int((xlim[1] - xlim[0])/0.1) + 1)],
        ytick=[x * 0.1 + ymin for x in
               range(int((ylim[1] - ylim[0])/0.1) + 1)],
        xtick_size=17 * rate,
        ytick_size=17 * rate,
        linewidth=4 * rate,
        minor_xtick_num=2, minor_ytick_num=2,
        return_figure=True)
    ax1.plot(
        cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=2*rate, label=None)
    ax1.plot(
        (cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
        '-k', lw=2*rate, label=None)
    ax1.plot(
        outer_gamut[:, 0], outer_gamut[:, 1], '--k',
        label=cs.BT709, lw=1.0*rate)
    ax1.scatter(
        ref_xyY[..., 0], ref_xyY[..., 1], marker='o', s=150*rate, c='none',
        edgecolor=marker_color.reshape(rgbmyc_num + color_checker_num, 3),
        lw=3*rate)
    ax1.scatter(
        xyY[..., 0], xyY[..., 1], marker='x', s=100*rate, lw=2*rate,
        c='black')

    # dummy plot (for legend)
    ax1.scatter(
        [1], [1], marker='o', s=150*rate, label="Reference value",
        edgecolor='k', c='none')
    ax1.scatter(
        [1], [1], marker='x', s=100*rate, label="Measured value", lw=2*rate,
        c='black')

    pu.show_and_save(fig=fig, legend_loc='upper right', save_fname=graph_name)


def plot_gamut_cms_result_with_in_out(
        data_dict, in_gamut_name, in_gamma_name,
        out_gamut_name, out_gamma_name, graph_name):
    rgbmyc_num = 6
    color_checker_num = 19
    rate = 1.1

    data = np.hstack(
        [data_dict['rgbmyc'],
         data_dict['colorchecker'][:, :color_checker_num]])
    linear_rgb = tf.eotf(data / 255, out_gamma_name)
    large_xyz = cs.calc_XYZ_from_rgb(
        rgb=linear_rgb, color_space_name=out_gamut_name)
    xyY = XYZ_to_xyY(large_xyz)

    rgbmyc_ref_rgb = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1]])
    rgbmyc_ref_rgb = tf.eotf(rgbmyc_ref_rgb, in_gamma_name)
    rgbmyc_ref_rgb = RGB_to_RGB(
        rgbmyc_ref_rgb,
        RGB_COLOURSPACES[in_gamut_name], RGB_COLOURSPACES[out_gamut_name])
    color_checker_ref_rgb = tpg.generate_color_checker_rgb_value(
        color_space=RGB_COLOURSPACES[out_gamut_name], target_white=cs.D65)
    ref_rgb = np.hstack(
        [rgbmyc_ref_rgb.reshape(1, rgbmyc_num, 3),
         color_checker_ref_rgb[:color_checker_num].reshape(
             1, color_checker_num, 3)])
    ref_xyz = cs.calc_XYZ_from_rgb(
        rgb=ref_rgb, color_space_name=out_gamut_name)
    ref_xyY = XYZ_to_xyY(ref_xyz)

    marker_color = RGB_to_RGB(
        ref_rgb,
        RGB_COLOURSPACES[out_gamut_name], RGB_COLOURSPACES[cs.BT709])
    marker_color = tf.oetf(np.clip(marker_color, 0.0, 1.0), tf.GAMMA24)

    # diagram
    xmin = 0.0
    xmax = 0.8
    ymin = 0.0
    ymax = 0.9
    cmf_xy = tpg._get_cmfs_xy()
    xlim = (min(0, xmin), max(0.8, xmax))
    ylim = (min(0, ymin), max(0.9, ymax))
    figsize_h = 8 * rate
    figsize_v = 9 * rate
    # gamut の用意
    outer_gamut, _ = tpg.get_primaries(cs.BT709)
    fig, ax1 = pu.plot_1_graph(
        bg_color=(0.8, 0.8, 0.8),
        fontsize=20 * rate,
        figsize=(figsize_h, figsize_v),
        graph_title="Color Checker and RGBMYC",
        xlabel=None, ylabel=None,
        legend_size=18 * rate,
        xlim=xlim, ylim=ylim,
        xtick=[x * 0.1 + xmin for x in
               range(int((xlim[1] - xlim[0])/0.1) + 1)],
        ytick=[x * 0.1 + ymin for x in
               range(int((ylim[1] - ylim[0])/0.1) + 1)],
        xtick_size=17 * rate,
        ytick_size=17 * rate,
        linewidth=4 * rate,
        minor_xtick_num=2, minor_ytick_num=2,
        return_figure=True)
    ax1.plot(
        cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=2*rate, label=None)
    ax1.plot(
        (cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
        '-k', lw=2*rate, label=None)
    ax1.plot(
        outer_gamut[:, 0], outer_gamut[:, 1], '--k',
        label=cs.BT709, lw=1.0*rate)
    ax1.scatter(
        ref_xyY[..., 0], ref_xyY[..., 1], marker='o', s=150*rate, c='none',
        edgecolor=marker_color.reshape(rgbmyc_num + color_checker_num, 3),
        lw=3*rate)
    ax1.scatter(
        xyY[..., 0], xyY[..., 1], marker='x', s=100*rate, lw=2*rate,
        c='black')

    # dummy plot (for legend)
    ax1.scatter(
        [1], [1], marker='o', s=150*rate, label="Reference value",
        edgecolor='k', c='none')
    ax1.scatter(
        [1], [1], marker='x', s=100*rate, label="Measured value", lw=2*rate,
        c='black')

    pu.show_and_save(fig=fig, legend_loc='upper right', save_fname=graph_name)


def concat_result_one_file(
        tf_graph_name, gamut_graph_name, concat_name,
        in_gamut_name, out_gamut_name, in_gamma_name, out_gamma_name, env):
    width = 1920
    height = 1080
    font = fc1.NOTO_SANS_CJKJP_BLACK
    font_size = 40
    font_edge_size = 6
    font_edge_color = (192, 192, 192)
    text = f" {env}, {in_gamut_name}-{in_gamma_name}, "
    text += f"{out_gamut_name}-{out_gamma_name}"

    img = np.zeros((height, width, 3))
    _, text_height = fc1.get_text_size(
        text=text, font_size=font_size, font_path=font,
        stroke_width=font_edge_size, stroke_fill=font_edge_color)
    text_pos = [0, 0 + int(text_height * 0.1)]
    text_drawer = fc1.TextDrawer(
        img=img, text=text, pos=text_pos,
        font_color=(16/255, 16/255, 16/255), font_size=font_size,
        font_path=font,
        stroke_width=font_edge_size, stroke_fill=font_edge_color)
    text_drawer.draw()
    # img = np.uint8(np.round(img * 255))

    # merge tf_graph
    tf_img = tpg.img_read(tf_graph_name) / 255
    tf_height, tf_width, _ = tf_img.shape
    tpg.merge(
        img, tf_img, [width//4 - tf_width//2, height//2 - tf_height//2])

    # merge gamut_graph
    gamut_img = tpg.img_read(gamut_graph_name) / 255
    gamut_height, gamut_width, _ = gamut_img.shape
    tpg.merge(
        img, gamut_img,
        [width//2 + width//4 - gamut_width//2, height//2 - gamut_height//2])

    img = np.uint8(np.round(img * 255))

    tpg.img_write(concat_name, img)


def analyze_browser_cms_result(
        fname, gamut_name=cs.BT709, gamma_name=tf.GAMMA24, env="Win 11"):
    tf_graph_name = create_result_graph_name(fname, suffix='_tf')
    gamut_graph_name = create_result_graph_name(fname, suffix='_gamut')
    concat_name = create_result_graph_name(
        fname, suffix='_gamut', output_dir="./concat_result")

    data_dict = read_code_value_from_gradation_pattern(fname)
    plot_transfer_characteristics_cms_result(
        data=data_dict['ramp'], graph_name=tf_graph_name)
    plot_gamut_cms_result(
        data_dict=data_dict,
        gamut_name=gamut_name, gamma_name=gamma_name,
        graph_name=gamut_graph_name)
    concat_result_one_file(
        tf_graph_name=tf_graph_name, gamut_graph_name=gamut_graph_name,
        concat_name=concat_name, gamut_name=gamut_name,
        gamma_name=gamma_name, env=env)


def analyze_browser_cms_result_with_in_out(
        fname, in_gamut_name=cs.BT709, out_gamut_name=cs.BT709,
        in_gamma_name=tf.SRGB, out_gamma_name=tf.SRGB, env="Win 11"):
    tf_graph_name = create_result_graph_name(fname, suffix='_tf')
    gamut_graph_name = create_result_graph_name(fname, suffix='_gamut')
    concat_name = create_result_graph_name(
        fname, suffix='_gamut', output_dir="./concat_result")

    data_dict = read_code_value_from_gradation_pattern(fname)
    plot_transfer_characteristics_cms_result_with_in_out(
        data=data_dict['ramp'],
        in_tf=in_gamma_name, out_tf=out_gamma_name, graph_name=tf_graph_name)
    plot_gamut_cms_result_with_in_out(
        data_dict=data_dict,
        in_gamut_name=in_gamut_name, in_gamma_name=in_gamma_name,
        out_gamut_name=out_gamut_name, out_gamma_name=out_gamma_name,
        graph_name=gamut_graph_name)
    concat_result_one_file(
        tf_graph_name=tf_graph_name, gamut_graph_name=gamut_graph_name,
        concat_name=concat_name,
        in_gamut_name=in_gamut_name, out_gamut_name=out_gamut_name,
        in_gamma_name=in_gamma_name, out_gamma_name=out_gamma_name, env=env)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
    # debug_func()
