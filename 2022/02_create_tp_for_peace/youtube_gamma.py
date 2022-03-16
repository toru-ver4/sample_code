# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
import subprocess
from pathlib import Path
from turtle import color

# import third-party libraries
import numpy as np
from colour.models import BT709_COLOURSPACE, RGB_COLOURSPACES
from colour import XYZ_to_xyY
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


def _func_t_inverse(t):
    upper = (t > const_lab_delta) * (t ** 3)
    lower = (t <= const_lab_delta) * 3 * (const_lab_delta ** 2) * (t - 4/29)
    return upper + lower


def create_debug_img_with_icc_profile():
    fname = "./img/src_bt709_gm24.png"
    fname_with_profile = "./img/src_bt709_gm24_with_profile.png"
    icc_profile = "./icc_profile/Gamma2.4_BT.709_D65.icc"
    cmd = [
        'convert', fname, '-define', "png:color-type=2",
        '-profile', icc_profile, fname_with_profile]
    subprocess.run(cmd)


def main_func():
    # create_youtube_srgb_gm24_pattern()
    create_debug_img_with_icc_profile()

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


def debug_func():
    # plot_srgb_gm24_oetf()
    plot_srgb_to_gm24()


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


def read_code_value_from_gradation_pattern(
        fname, gamut_name=cs.BT709, gamma_name=tf.GAMMA24):
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


def plot_gamut_cms_result(data_dict, gamut_name, gamma_name, graph_name):
    rgbmyc_num = 6
    color_checker_num = 19
    rate = 1.1

    data = np.hstack(
        [data_dict['rgbmyc'],
         data_dict['colorchecker'][:, :color_checker_num]])
    linear_rgb = tf.eotf(data / 255, gamma_name)
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
    outer_gamut, _ = tpg.get_primaries(gamut_name)
    fig, ax1 = pu.plot_1_graph(
        bg_color=(0.8, 0.8, 0.8),
        fontsize=20 * rate,
        figsize=(figsize_h, figsize_v),
        graph_title="CIE1931 Chromaticity Diagram",
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
        label=gamut_name, lw=1.0*rate)
    ax1.scatter(
        ref_xyY[..., 0], ref_xyY[..., 1], marker='o', s=150*rate, c='none',
        edgecolor=data.reshape(rgbmyc_num + color_checker_num, 3)/255,
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
        gamut_name, gamma_name, env):
    width = 1920
    height = 1080
    font = fc1.NOTO_SANS_CJKJP_BLACK
    font_size = 40
    font_edge_size = 6
    font_edge_color = (192, 192, 192)
    text = f' {env}, {gamma_name}, {gamut_name}'

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

    data_dict = read_code_value_from_gradation_pattern(
        fname, gamut_name=gamut_name, gamma_name=gamma_name)
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


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # main_func()
    debug_func()
