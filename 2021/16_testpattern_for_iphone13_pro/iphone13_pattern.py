# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
import subprocess
from itertools import product
from math import ceil

# import third-party libraries
import numpy as np
import cv2
from colour import RGB_to_RGB, RGB_COLOURSPACES, RGB_to_XYZ, XYZ_to_xyY

# import my libraries
import test_pattern_generator2 as tpg
import ty_utility as util
import font_control as fc
import color_space as cs
import transfer_functions as tf
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def modify_to_n_times(x, n):
    if x % n != 0:
        y = int(ceil(x / n)) * n
    else:
        y = x

    return y


def dot_mesh_pattern(
        width_org=1920, height_org=1080, dot_size=2, color=[1, 1, 0]):

    width = modify_to_n_times(width_org, dot_size * 2)
    height = modify_to_n_times(height_org, dot_size * 2)
    img = np.ones((height, width, 3)) * np.array(color)

    fname = f"./img/{width_org}x{height_org}_dotsize-{dot_size}_rgbmask-"
    fname += f"{color[0]}{color[1]}{color[2]}.png"

    zero_idx_h = ((np.arange(width) // (2**(dot_size-1))) % 2) == 0
    idx_even = np.hstack([zero_idx_h for x in range(dot_size)])
    idx_odd = np.hstack([~zero_idx_h for x in range(dot_size)])
    idx_even_odd = np.hstack([idx_even, idx_odd])
    idx_all_line = np.tile(
        idx_even_odd, height//(2 * dot_size)).reshape(height, width)

    img[idx_all_line] = 0

    img = img[:height_org, :width_org]
    print(fname)
    tpg.img_wirte_float_as_16bit_int(fname, img)

    fname_icc = util.add_suffix_to_filename(fname=fname, suffix="_with_icc")
    icc_profile = './icc_profile/Gamma2.4_DCI-P3_D65.icc'
    cmd = ['convert', fname, '-profile', icc_profile, fname_icc]
    subprocess.run(cmd)


def create_dot_pattern():
    resolution_list = [[2778, 1284], [2532, 1170]]
    dot_size_list = [1, 2]
    color_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    for resolution, dot_size, color in product(
            resolution_list, dot_size_list, color_list):
        width = resolution[0]
        height = resolution[1]
        dot_mesh_pattern(
            width_org=width, height_org=height, dot_size=dot_size, color=color)


def create_abl_check_pattern(width_panel=2778, height_panel=1284):
    fps = 60
    sec = 8
    frame = fps * sec
    width_total = 1920
    height_total = 1080
    width = width_total
    height = int(round(height_panel/width_panel * width))

    for idx in range(frame):
        rate = (np.sin(np.pi/(frame - 1)*idx - np.pi/2) + 1) / 2
        img = np.zeros((height_total, width_total, 3))

        local_width = int(round(width * rate))
        local_height = int(round(height * rate))
        st_pos = (
            (width_total//2) - (local_width//2),
            (height_total//2) - (local_height//2))
        ed_pos = (st_pos[0]+local_width, st_pos[1]+local_height)
        cv2.rectangle(img, st_pos, ed_pos, (1.0, 1.0, 1.0), -1)

        percent = (local_width * local_height)\
            / (width * height) * 100
        text_drawer = fc.TextDrawer(
            img, text=f"{percent:.02f}%",
            pos=(int(width_total*0.04), int(width_total*0.08)),
            font_color=(0.25, 0.25, 0.25), font_size=30)
        text_drawer.draw()

        fname = "/work/overuse/2021/00_iphone_movie/img_seq/"
        fname += f"iPhone_abl_{width_panel}x{height_panel}_{idx:04d}.png"
        print(fname)
        tpg.img_wirte_float_as_16bit_int(fname, img)


def create_patch_specific_area(
        panel_width=2778, panel_height=1284, area_rate=20.0,
        color_st2084=[0.8, 0.5, 0.2], luminance=1000, src_cs=cs.BT709):
    width = 1920
    height = 1080
    tf_str = tf.ST2084
    img = np.zeros((height, width, 3))
    width_vertual = panel_width/panel_height*height
    height_vertual = height
    block_size = int(
        round((area_rate/100 * width_vertual * height_vertual) ** 0.5))
    st_pos = ((width//2) - (block_size//2), (height//2) - (block_size//2))
    ed_pos = (st_pos[0]+block_size, st_pos[1]+block_size)

    color_with_lumiannce = calc_linear_color_from_primary(
        color=color_st2084, luminance=luminance)
    cv2.rectangle(img, st_pos, ed_pos, color_with_lumiannce, -1)

    large_xyz = RGB_to_XYZ(
        color_with_lumiannce, cs.D65, cs.D65,
        RGB_COLOURSPACES[src_cs].matrix_RGB_to_XYZ)
    xyY = XYZ_to_xyY(large_xyz)

    img = tf.oetf(np.clip(img, 0.0, 1.0), tf_str)

    text = f"for_{panel_width}x{panel_height}, {src_cs}, "
    text += f"xyY=({xyY[0]:.03f}, "
    text += f"{xyY[1]:.03f}, {xyY[2]*10000:.1f})"
    text_drawer = fc.TextDrawer(
        img, text=text, pos=(10, 10),
        font_color=(0.25, 0.25, 0.25), font_size=20)
    text_drawer.draw()

    fname = f"./img/iPhone13_color_patch_for_{panel_width}x{panel_height}_"
    fname += f"{src_cs}_"
    fname += f"rgb_{color_with_lumiannce[0]:.2f}-"
    fname += f"{color_with_lumiannce[1]:.2f}-"
    fname += f"{color_with_lumiannce[2]:.2f}_{tf_str}_{luminance}-nits.png"
    print(fname)
    tpg.img_wirte_float_as_16bit_int(fname, img)


def calc_linear_color_from_primary(color=[1, 0, 0], luminance=1000):
    """
    supported transfer characteristics is ST2084 only.
    """
    color_luminance\
        = tf.eotf(np.array(color), tf.ST2084)\
        / tf.PEAK_LUMINANCE[tf.ST2084] * luminance

    return color_luminance


def create_iphone_13_primary_patch(area_rate=0.4*100):
    color_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    resolution_list = [[2778, 1284], [2532, 1170]]
    luminance_list = [100, 1000, 4000, 10000]
    src_cs_list = [cs.BT709, cs.P3_D65, cs.BT2020]

    for resolution, luminance, color, src_cs in product(
            resolution_list, luminance_list, color_list, src_cs_list):
        width = resolution[0]
        height = resolution[1]
        create_patch_specific_area(
            panel_width=width, panel_height=height, area_rate=area_rate,
            color_st2084=color, luminance=luminance, src_cs=src_cs)


def conv_img_from_bt2020_to_bt709_using_3x3_matrix():
    # in_fname = "./img/bt2020_bt709_hue_chroma_1920x1080_h_num-32.png"
    in_fname = "./img/iPhone13_color_patch_for_2778x1284_P3-D65-on-ITU-R "
    in_fname += "BT.2020_rgb_0.10-0.00-0.00_SMPTE ST2084_1000-nits.png"
    out_fname = util.add_suffix_to_filename(
        fname=in_fname, suffix="_bt709_with_matrix")
    tf_str = tf.ST2084

    img_non_linear = tpg.img_read_as_float(in_fname)
    img_linear_2020 = tf.eotf(img_non_linear, tf_str)
    img_linear_709 = RGB_to_RGB(
        RGB=img_linear_2020,
        input_colourspace=RGB_COLOURSPACES[cs.BT2020],
        output_colourspace=RGB_COLOURSPACES[cs.P3_D65])
    img_non_linear_709 = tf.oetf(np.clip(img_linear_709, 0.0, 1.0), tf_str)

    tpg.img_wirte_float_as_16bit_int(out_fname, img_non_linear_709)


def plot_bt2020_vs_dci_p3():
    bt2020 = tpg.get_primaries(cs.BT2020)[0]
    p3_d65 = tpg.get_primaries(cs.P3_D65)[0]
    cmf_xy = tpg._get_cmfs_xy()

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(8, 9),
        bg_color=(0.90, 0.90, 0.90),
        graph_title="Chromaticity Diagram",
        graph_title_size=None,
        xlabel="x", ylabel="y",
        axis_label_size=None,
        legend_size=17,
        xlim=[0.65, 0.72],
        ylim=[0.28, 0.34],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=4)
    ax1.plot(
        (cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
        '-k', lw=4)
    ax1.plot(
        bt2020[..., 0], bt2020[..., 1], '-', color=pu.RED, label="BT.2020")
    ax1.plot(
        p3_d65[..., 0], p3_d65[..., 1], '-', color=pu.SKY, label="DCI-P3")
    pu.show_and_save(
        fig=fig, legend_loc='upper right', save_fname="./img/p3_gamut.png")


def create_gray_patch_core(
        width, height, panel_width, panel_height,
        st_pos, ed_pos, cv_float):
    img = np.zeros((height, width, 3))
    color = [cv_float, cv_float, cv_float]
    cv2.rectangle(img, st_pos, ed_pos, color, -1)

    text = f"for_{panel_width}x{panel_height}, {cv_float*1023:.0f} CV"
    text_drawer = fc.TextDrawer(
        img, text=text, pos=(10, 10),
        font_color=(0.25, 0.25, 0.25), font_size=20)
    text_drawer.draw()

    fname = f"./img/iPhone13_color_patch_for_{panel_width}x{panel_height}_"
    fname += f"{int(cv_float*1023):04d}-CV.png"
    print(fname)
    tpg.img_wirte_float_as_16bit_int(fname, img)


def create_gray_patch(panel_width=2778, panel_height=1284, area_rate=0.4*100):
    width = 1920
    height = 1080
    step_num = 33  # 33 or 65
    each_step = 1024 // (step_num - 1)
    cv_list = np.arange(step_num) * each_step
    cv_list[-1] = 1023

    width_vertual = panel_width/panel_height*height
    height_vertual = height
    block_size = int(
        round((area_rate/100 * width_vertual * height_vertual) ** 0.5))
    st_pos = ((width//2) - (block_size//2), (height//2) - (block_size//2))
    ed_pos = (st_pos[0]+block_size, st_pos[1]+block_size)

    for cv in cv_list:
        create_gray_patch_core(
            width=width, height=height,
            panel_width=panel_width, panel_height=panel_height,
            st_pos=st_pos, ed_pos=ed_pos, cv_float=cv/1023)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_dot_pattern()
    create_abl_check_pattern(width_panel=2778, height_panel=1284)
    create_abl_check_pattern(width_panel=2532, height_panel=1170)
    # create_patch_specific_area(
    #     panel_width=2778, panel_height=1284, area_rate=0.4*100,
    #     color_linear=[1, 0, 0],
    #     luminance=1000, src_cs=cs.BT709, dst_cs=cs.BT2020, tf_str=tf.ST2084)
    # create_iphone_13_primary_patch(area_rate=0.4*100)
    # conv_img_from_bt2020_to_bt709_using_3x3_matrix()
    # plot_bt2020_vs_dci_p3()
    # create_gray_patch(panel_width=2778, panel_height=1284, area_rate=0.4*100)
