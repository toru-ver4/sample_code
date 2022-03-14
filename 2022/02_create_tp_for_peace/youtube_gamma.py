# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from turtle import heading
from colour import sRGB_to_XYZ, write_image
import subprocess

# import third-party libraries
import numpy as np

# import my libraries
import plot_utility as pu
import test_pattern_generator2 as tpg
import transfer_functions as tf
import font_control as fc1

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


FFMPEG_NORMALIZE_COEF = 65340
const_lab_delta = 6.0/29.0


def _func_t_inverse(t):
    upper = (t > const_lab_delta) * (t ** 3)
    lower = (t <= const_lab_delta) * 3 * (const_lab_delta ** 2) * (t - 4/29)
    return upper + lower


def main_func():
    create_youtube_srgb_gm24_pattern()


def debug_func():
    plot_srgb_gm24_oetf()


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
    lstar = _func_t_inverse((x * 100 + 16)/116) * 100
    pq = tf.eotf_to_luminance(x, tf.ST2084)
    x_max = 40
    x_min = -x_max * 0.04
    x_tick_unit = 4
    x_tick_num = (x_max // 4) + 1
    y_max = 1.6
    y_min = -y_max * 0.04

    h_line_list = []
    for idx in range(x_max + 1):
        y_val = pq[idx]
        points = [[x_min, x_max], [y_val, y_val]]
        h_line_list.append(points)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(14, 12),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Title",
        graph_title_size=None,
        xlabel="Code Value (8 bit)",
        ylabel="Luminance [cd/m2]",
        axis_label_size=None,
        legend_size=17,
        xlim=[x_min, x_max],
        ylim=[y_min, y_max],
        xtick=[x * x_tick_unit for x in range(x_tick_num)],
        ytick=[0, 1],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.yaxis.grid(None)
    ax1.plot(x_8bit, srgb, '-o', color=pu.RED, label="sRGB (CR 1000:1)")
    ax1.plot(x_8bit, gm24, '-o', color=pu.SKY, label="Gamma 2.4 (CR 1000:1)")
    ax1.plot(x_8bit, lstar, '-o', color=pu.BROWN, label="CIE 1976 L*")
    ax1.plot(x_8bit, pq, '-o', color=pu.MAJENTA, label="SMPTE ST 2084")
    for points in h_line_list:
        ax1.plot(points[0], points[1], '--', lw=0.5, color='k')
    pu.show_and_save(
        fig=fig, legend_loc='upper left',
        save_fname="./img/srgb_vs_gm24.png")


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


def create_alpha_black(alpha=128):
    width = 940
    height = 1080
    img = np.ones((height, width, 4), dtype=np.uint8)
    img[:, :, :3] = 0
    img[:, :, 3] = alpha

    fname = f"./img/bg_alpha-{alpha}.png"
    tpg.img_write(fname, img)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # main_func()
    # debug_func()
    create_alpha_black(64)
    create_alpha_black(96)
    create_alpha_black(128)
    create_alpha_black(168)
    create_alpha_black(192)
