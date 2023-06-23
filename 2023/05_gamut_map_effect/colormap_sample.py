# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import XYZ_to_Oklab, xy_to_XYZ, LCHab_to_Lab
from colour.utilities import tstack

# import my libraries
import plot_utility as pu
import color_space as cs
import transfer_functions as tf
from color_space_plot import create_valid_oklab_cl_plane_image_gm24
from test_pattern_coordinate import GridCoordinate
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2023 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

color_info_list = [
    {'st_cl': [0.05, 0.895], 'ed_cl': [0.25, 0.50],
     'hue_val': 263.78, 'c_name': 'B'},
    # {'st_cl': [0.05, 0.895], 'ed_cl': [0.15, 0.90],
    #  'hue_val': 194.81, 'c_name': 'C'},
    {'st_cl': [0.05, 0.55], 'ed_cl': [0.25, 0.85],
     'hue_val': 142.51, 'c_name': 'G'},
    {'st_cl': [0.05, 0.55], 'ed_cl': [0.20, 0.95],
     'hue_val': 109.79, 'c_name': 'Y'},
    {'st_cl': [0.05, 0.896], 'ed_cl': [0.24, 0.6],
     'hue_val': 29.22, 'c_name': 'R'},
    {'st_cl': [0.05, 0.35], 'ed_cl': [0.25, 0.75],
     'hue_val': 328.35, 'c_name': 'M'},
]


def debug_plot_oklab():
    large_xyz = xy_to_XYZ(cs.D65)
    oklab = XYZ_to_Oklab(large_xyz)
    print(oklab)

    primary_rgb = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    primary_xyz = cs.rgb_to_large_xyz(primary_rgb, cs.BT2020)
    primary_oklab = XYZ_to_Oklab(primary_xyz)
    print(primary_oklab)
    primary_oklch = cs.oklab_to_oklch(primary_oklab)
    print(primary_oklch)
    primary_oklab_back = cs.oklch_to_oklab(primary_oklch)
    print(primary_oklab_back)

    oklab = cs.rgb_to_oklab(primary_rgb, cs.BT2020)
    print(oklab)
    print(cs.oklab_to_oklch(oklab))
    rgb = cs.oklab_to_rgb(oklab, cs.BT2020)
    print(rgb)


def check_primary_secondary_hue_angle(ps_color):
    xyz = cs.rgb_to_large_xyz(ps_color, cs.BT709)
    oklab = cs.XYZ_to_Oklab(xyz)
    oklch = cs.oklab_to_oklch(oklab)
    # print(oklch[..., 2])

    return oklch[..., 2]


def plot_primary_secondary_okcl_single_plane(
        hue_angle=0, color_name="R", color_space_name=cs.BT709):
    # bg_lut = TyLchLut(lut=np.load(bg_lut_name))
    h_val = hue_angle
    sample_num = 2048
    # jj_sample = 2048
    ll_max = 1.0
    cc_max = 0.4
    print(f"h_val={h_val} started")

    rgb_gm24 = create_valid_oklab_cl_plane_image_gm24(
        h_val=h_val, c_max=cc_max, c_sample=sample_num, l_sample=sample_num,
        color_space_name=color_space_name, bg_val=0.5)
    graph_title = f"CL plane,  {color_space_name},  Hue={h_val:.2f}°,  "

    # ll_base = np.linspace(0, bg_lut.ll_max, jj_sample)
    # hh_base = np.ones_like(ll_base) * h_val
    # lh_array = tstack([ll_base, hh_base])
    # lch = bg_lut.interpolate(lh_array=lh_array)

    # chroma = lch[..., 1]
    # lightness = lch[..., 0]

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        bg_color=(0.96, 0.96, 0.96),
        figsize=(12, 12),
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
    # ax1.plot(chroma, lightness, color='k')
    fname = f"./img/oklab_lch_plane_{color_name}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)


def plot_primary_secondary_okcl_single_plane_with_map_line(
        color_space_name=cs.BT709, color_info=None):
    h_val = color_info['hue_val']
    color_name = color_info['c_name']
    sample_num = 2048
    ll_max = 1.0
    cc_max = 0.35
    print(f"h_val={h_val} started")

    x_pos = [color_info['st_cl'][0], color_info['ed_cl'][0]]
    y_pos = [color_info['st_cl'][1], color_info['ed_cl'][1]]
    rgb_gm24 = create_valid_oklab_cl_plane_image_gm24(
        h_val=h_val, c_max=cc_max, c_sample=sample_num, l_sample=sample_num,
        color_space_name=color_space_name, bg_val=0.2)
    graph_title = f"CL plane,  {color_space_name},  Hue={h_val:.2f}°,  "

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        bg_color=(0.96, 0.96, 0.96),
        figsize=(12, 12),
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
    ax1.plot(
        x_pos, y_pos, 'o--', color='k')
    fname = f"./img/oklab_lch_plane_with_line_{color_name}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)


def plot_primary_secondary_okcl_planes():
    ps_color = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1],
         [1, 0, 1], [1, 1, 0], [0, 1, 1]])
    ps_name = ['R', "G", "B", "M", "Y", "C"]
    hue_angles = check_primary_secondary_hue_angle(ps_color=ps_color)
    for color_name, hue_angle in zip(ps_name, hue_angles):
        plot_primary_secondary_okcl_single_plane(
            hue_angle=hue_angle, color_name=color_name)


def debug_color_scale():
    for color_info in color_info_list:
        plot_primary_secondary_okcl_single_plane_with_map_line(
            color_info=color_info)


def create_scale_image(h_val, x_pos, y_pos, width, height):
    x = np.linspace(0, 1, width)
    cc = (x_pos[1] - x_pos[0]) * x + x_pos[0]
    ll = (y_pos[1] - y_pos[0]) * x + y_pos[0]
    h_pos = np.ones_like(ll) * h_val
    lc_pos = tstack([ll, cc, h_pos])
    print(lc_pos)
    oklab = cs.oklch_to_oklab(lc_pos)
    rgb = cs.oklab_to_rgb(oklab=oklab, color_space_name=cs.BT709)
    img = tpg.h_color_line_to_img(line=rgb, height=height)

    return img


def debug_plot_color_scale(color_info_list=None):
    v_num = len(color_info_list)
    bg_width = 1920
    bg_height = 1080
    fg_width = 1680
    fg_height = 150
    bg_color = 0.01

    img = np.ones((bg_height, bg_width, 3)) * bg_color
    gc = GridCoordinate(
        bg_width=bg_width, bg_height=bg_height,
        fg_width=fg_height, fg_height=fg_height, h_num=1, v_num=v_num)
    st_pos_list = gc.get_st_pos_list()
    st_pos_list[..., 0] = (bg_width // 2) - (fg_width // 2)
    st_pos_list = st_pos_list[0]
    for idx, color_info in enumerate(color_info_list):
        h_val = color_info['hue_val']
        color_name = color_info['c_name']
        x_pos = [color_info['st_cl'][0], color_info['ed_cl'][0]]
        y_pos = [color_info['st_cl'][1], color_info['ed_cl'][1]]
        # print(x_pos, y_pos, h_val)
        # print(oklab)
        temp_img = create_scale_image(
            h_val=h_val, x_pos=x_pos, y_pos=y_pos,
            width=fg_width, height=fg_height)
        st_pos = st_pos_list[idx]
        ed_pos = np.zeros_like(st_pos)
        ed_pos[0] = st_pos[0] + fg_width
        ed_pos[1] = st_pos[1] + fg_height
        img[st_pos[1]:ed_pos[1], st_pos[0]:ed_pos[0]] = temp_img

    fname = "./img/color_maps.png"
    img = img ** (1/2.4)
    tpg.img_wirte_float_as_16bit_int(fname, img)


def make_color_map_lut_name():
    return "./color_map_lut.npy"


def create_color_map_lut(color_info_list):
    num_of_elem = 256
    num_of_color = len(color_info_list)
    num_of_rgb = 3
    lut_name = make_color_map_lut_name()
    luts = np.zeros((num_of_color, num_of_elem, num_of_rgb))

    for c_idx in range(num_of_color):
        color_info = color_info_list[c_idx]
        hue = color_info['hue_val']
        st_cl = color_info['st_cl']
        ed_cl = color_info['ed_cl']
        cc = np.linspace(st_cl[0], ed_cl[0], num_of_elem)
        ll = np.linspace(st_cl[1], ed_cl[1], num_of_elem)
        hh = np.ones_like(cc) * hue
        lch = tstack([ll, cc, hh])
        oklab = cs.oklch_to_oklab(lch)
        rgb = cs.oklab_to_rgb(oklab=oklab, color_space_name=cs.BT709)
        luts[c_idx] = rgb

    luts = luts ** (1/2.4)

    np.save(lut_name, luts)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # debug_plot_oklab()
    # ps_color = np.array(
    #     [[1, 0, 0], [0, 1, 0], [0, 0, 1],
    #      [1, 0, 1], [1, 1, 0], [0, 1, 1]])
    # print(check_primary_secondary_hue_angle(ps_color))
    # plot_primary_secondary_okcl_planes()
    debug_color_scale()
    # debug_plot_color_scale(color_info_list=color_info_list)
    create_color_map_lut(color_info_list=color_info_list)
