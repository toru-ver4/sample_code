# -*- coding: utf-8 -*-
"""

==========

"""

# import standard libraries
import os

# import third-party libraries
from colour.utilities import tstack
from colour import XYZ_to_xy, xy_to_XYZ, RGB_to_RGB
import numpy as np

# import my libraries
import test_pattern_generator2 as tpg
from test_pattern_coordinate import GridCoordinate
import transfer_functions as tf
import font_control2 as fc2
import color_space as cs
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def cut_two_points_into_N_points(pos1, pos2, n):
    """
    Parameters
    ----------
    pos1 : list
        stating point. [x1, y1]
    pos2 : list
        ending point. [x2, y2]
    n : int
        division number

    Returns
    -------
    ndarray
        position list

    Examples
    --------
    >>> pos1 = 0.5, 0.5
    >>> pos2 = 1.0, 0.7
    >>> pos_list = cut_two_points_into_N_points(pos1, pos2, 3)
    >>> print(pos_list)
    [[ 0.5   0.5 ]
     [ 0.75  0.6 ]
     [ 1.    0.7 ]]
    """
    x1, y1 = pos1
    x2, y2 = pos2
    x = np.linspace(x1, x2, n)
    y = np.linspace(y1, y2, n)
    pos_list = tstack([x, y])

    return pos_list


def calc_info_area_height(height, height_rate):
    return int(height * height_rate + 0.5)


def search_font_size(
        width, height, text, font, padding_rate_v=0.45, stroke_width=0):
    """
    Parameters
    ----------
    width : int
        A width of the draw area
    height : int
        A height of the draw area
    text : str
        text
    font : str
        font name
    padding_rate : float
        A padding ratio.
    stroke_width : int
        A brink width

    Retuns
    ------
    int
        adjusted font size
    """
    padding_rate_h = 0.9
    init_font_size = 5
    font_size = init_font_size
    num_of_loop = 196
    max_width = int(width * padding_rate_h + 0.5)
    max_height = int(height * padding_rate_v + 0.5)

    for idx in range(96):
        text_draw_ctrl = fc2.TextDrawControl(
            text=text, font_color=[0.5, 0.5, 0.5],
            font_size=font_size, font_path=font,
            stroke_width=stroke_width, stroke_fill=None)
        text_width, text_height = text_draw_ctrl.get_text_width_height()
        # print(text_width, text_height)

        if (text_width < max_width) and (text_height < max_height):
            font_size = init_font_size + idx + 1
        else:
            break

    if font_size >= (init_font_size + num_of_loop):
        raise ValueError("TY unknown error in search_font_size.")

    print(f'[Info] Selected font size is "{font_size}"')
    return font_size


def carete_tp_bottom_information_image(
        width, height, text1="hoge", text2=None,
        height_rate=0.04, font_size='auto', font=fc2.NOTO_SANS_CJKJP_MEDIUM,
        font_color=[0.2, 0.2, 0.2], bg_color=[0.00, 0.00, 0.00]):
    """
    Parameters
    ----------
    width : int
        A image width of the test pattern.
    height : int
        A image height of the test pattern.
    text1 : str
        A left-justified text .
    text2 : str
        A right-justified text .
    height_rate : float
        A information area size rate.
        ex. height_rate = 0.05, info_height = int(height * 0.05 + 0.5)

    Returns
    -------
    ndarray
        infromation image
    """
    info_height = calc_info_area_height(height=height, height_rate=height_rate)
    img = np.ones((info_height, width, 3)) * np.array(bg_color)

    if font_size == "auto":
        font_size = search_font_size(
            width=width, height=info_height, text=text1, font=font)

    text_draw_ctrl = fc2.TextDrawControl(
        text=text1, font_color=font_color, font_size=font_size,
        font_path=font, stroke_width=0, stroke_fill=None)

    # calc position
    _, text_height = text_draw_ctrl.get_text_width_height()
    h_pos = (info_height - text_height) // 2
    v_pos = (info_height - text_height) // 2
    pos = (h_pos, v_pos)

    text_draw_ctrl.draw(img=img, pos=pos)

    if text2 is not None:
        text_draw_ctrl = fc2.TextDrawControl(
            text=text2, font_color=font_color, font_size=font_size,
            font_path=font, stroke_width=0, stroke_fill=None)
        # calc position
        text_width, text_height = text_draw_ctrl.get_text_width_height()
        h_pos = width - text_width - ((info_height - text_height) // 2)
        v_pos = (info_height - text_height) // 2
        pos = (h_pos, v_pos)
        text_draw_ctrl.draw(img=img, pos=pos)

    return img


def calc_6color_xy_coordinate(div_num=6):
    """
    Returns
    -------
    ndarray
        rgb list. shape is (num_of_color, div_num, 2).

    Examples
    --------
    >>> calc_6color_xy_coordinate(div_num=3)
    [[[ 0.3127      0.329     ]
      [ 0.51035     0.3105    ]
      [ 0.708       0.292     ]]

     [[ 0.3127      0.329     ]
      [ 0.379613    0.43321782]
      [ 0.44652599  0.53743564]]

     [[ 0.3127      0.329     ]
      [ 0.24135     0.563     ]
      [ 0.17        0.797     ]]

     [[ 0.3127      0.329     ]
      [ 0.22960212  0.33677795]
      [ 0.14650423  0.34455589]]

     [[ 0.3127      0.329     ]
      [ 0.22185     0.1875    ]
      [ 0.131       0.046     ]]

     [[ 0.3127      0.329     ]
      [ 0.34043019  0.23805585]
      [ 0.36816038  0.14711171]]]
    """
    white = cs.D65
    rygcbm_rgb = np.array([
        [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1],
        [0, 0, 1], [1, 0, 1]])
    large_xyz = cs.rgb_to_large_xyz(rygcbm_rgb, cs.BT2020)
    xy = XYZ_to_xy(large_xyz)

    num_of_color = len(xy)
    xy_list = np.zeros((num_of_color, div_num, 2))
    for idx, ed_pos in enumerate(xy):
        xy_one_color = cut_two_points_into_N_points(white, ed_pos, div_num)
        xy_list[idx] = xy_one_color

    return xy_list


def calc_6color_rgb_val(div_num=4):
    """
    Returns
    -------
    ndarray
        rgb list. shape is (num_of_color, div_num, 2).

    Examples
    --------
    >>> calc_6color_rgb_val(div_num=3)
    [[[  1.00000000e+00   1.00000000e+00   1.00000000e+00]
      [  1.00000000e+00   2.28387660e-01   2.28387660e-01]
      [  1.00000000e+00   2.91655824e-16   0.00000000e+00]]

     [[  1.00000000e+00   1.00000000e+00   1.00000000e+00]
      [  1.00000000e+00   1.00000000e+00   3.65427397e-01]
      [  1.00000000e+00   1.00000000e+00   1.82767356e-16]]

     [[  1.00000000e+00   1.00000000e+00   1.00000000e+00]
      [  2.18674457e-01   1.00000000e+00   2.18674457e-01]
      [  0.00000000e+00   1.00000000e+00   0.00000000e+00]]

     [[  1.00000000e+00   1.00000000e+00   1.00000000e+00]
      [  4.13149822e-01   1.00000000e+00   1.00000000e+00]
      [  0.00000000e+00   1.00000000e+00   1.00000000e+00]]

     [[  1.00000000e+00   1.00000000e+00   1.00000000e+00]
      [  2.97819965e-01   2.97819965e-01   1.00000000e+00]
      [  2.63352524e-17   1.05341010e-16   1.00000000e+00]]

     [[  1.00000000e+00   1.00000000e+00   1.00000000e+00]
      [  1.00000000e+00   4.18646481e-01   1.00000000e+00]
      [  1.00000000e+00   3.57493955e-16   1.00000000e+00]]]
    """
    xy = calc_6color_xy_coordinate(div_num=div_num)
    large_xyz = xy_to_XYZ(xy)
    rgb = cs.large_xyz_to_rgb(large_xyz, cs.BT2020)
    max_list = np.max(rgb, axis=-1)[:, :, np.newaxis]
    rgb = np.clip(rgb/max_list, 0, 1)

    return rgb


def make_patch_center_pos_list_fname(div_num):
    return f"./debug/patch_center_pos_div_num-{div_num}.npy"


def draw_patch(img, div_num=6):
    height, width = img.shape[:2]
    rgb_list = calc_6color_rgb_val(div_num=div_num)
    num_of_color, div_num = rgb_list.shape[:2]
    size = 150

    gc = GridCoordinate(
        bg_width=width, bg_height=height,
        fg_width=size, fg_height=size,
        h_num=num_of_color, v_num=div_num,
        remove_tblr_margin=False)
    pos_list = gc.get_st_pos_list()

    # save pos data for analyze
    pos_fnmae = make_patch_center_pos_list_fname(div_num)
    np.save(pos_fnmae, pos_list + (size // 2))

    for c_idx in range(num_of_color):
        for d_idx in range(div_num):
            st_pos = pos_list[c_idx, d_idx]
            ed_pos = [st_pos[0]+size, st_pos[1]+size]
            rgb = rgb_list[c_idx, d_idx]        
            img[st_pos[1]:ed_pos[1], st_pos[0]:ed_pos[0]] = rgb


def create_gamut_evaluation_pattern():
    width = 1920
    total_height = 1080
    info_area_height_rate = 0.04
    div_num = 6
    text1 = "WCG Pattern, EOTF: sRGB, Gamut: BT.2020, "
    text1 += f"White point: D65, {width}x{total_height}"
    text2 = "Rev.1"
    info_img = carete_tp_bottom_information_image(
        width=width, height=total_height,
        text1=text1, text2=text2,
        height_rate=info_area_height_rate)
    height = total_height - info_img.shape[0]
    img = np.ones((height, width, 3)) * np.array([0.01, 0.01, 0.01])
    draw_patch(img=img, div_num=div_num)

    img = np.vstack((img, info_img))

    img = tf.oetf(img, tf.SRGB)
    icc_profile_name = "../../2020/026_create_icc_profiles/sRGB_BT2020.icc"
    tpg.img_wirte_float_as_16bit_int_with_icc(
        filename="./img/test.png", img_float=img,
        icc_profile_name=icc_profile_name)


def get_result_rgb_value_from_img(img, div_num, tf_gamma):
    """
    Returns
    -------
    ndarray
        linear rgb list. shape is (num_of_color, div_nu, 3).
    """
    center_pos_fname = make_patch_center_pos_list_fname(div_num)
    center_pos = np.load(center_pos_fname)
    num_of_color = len(center_pos)

    rgb_list = np.zeros((num_of_color, div_num, 3))
    for c_idx in range(num_of_color):
        for d_idx in range(div_num):
            pos = center_pos[c_idx, d_idx]
            rgb = img[pos[1], pos[0]]
            rgb_list[c_idx, d_idx] = rgb

    rgb_list = tf.eotf_to_luminance(rgb_list, tf_gamma) / 100

    return rgb_list


def plot_result_pattern(result_img_fname, div_num, tf_gamma, gamut_name):
    """

    """
    result_img = tpg.img_read_as_float(result_img_fname)
    result_rgb = get_result_rgb_value_from_img(
        img=result_img, div_num=div_num, tf_gamma=tf_gamma)
    print(result_rgb)
    result_xy = XYZ_to_xy(cs.rgb_to_large_xyz(result_rgb, gamut_name))
    ref_xy = calc_6color_xy_coordinate(div_num=div_num)
    ref_rgb = tf.oetf(calc_6color_rgb_val(div_num=div_num), tf.SRGB)

    rate = 1.0
    fig, ax1 = pu.plot_chromaticity_diagram_base(
        rate=rate, bt709=True, p3d65=True, bt2020=True)
    ref_ms = 150
    measure_ms = 120
    ax1.scatter(
        [0.3127], [0.3290], s=ref_ms*rate, c='w', edgecolors='k',
        linewidths=1.5*rate, label="Theoretical value", zorder=0)
    ax1.scatter(
        [0.3127], [0.3290], s=measure_ms*rate, marker="X",
        linewidths=1.5*rate, label="Measured value", zorder=0,
        c='k', edgecolors='k')
    ax1.scatter(
        ref_xy[..., 0], ref_xy[..., 1],
        s=ref_ms*rate, c=ref_rgb.reshape(-1, 3), zorder=10,
        edgecolors='k', linewidth=1.5*rate)
    ax1.scatter(
        result_xy[..., 0], result_xy[..., 1], marker="X",
        s=measure_ms*rate, c=ref_rgb.reshape(-1, 3), zorder=10,
        linewidth=1.5*rate, edgecolors='k')
    pu.show_and_save(
        fig=fig, legend_loc='upper right',
        save_fname="./img/chromaticity_diagram_sample.png")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_gamut_evaluation_pattern()
    # plot_result_pattern(
    #     result_img_fname="./img/test.png", div_num=6,
    #     tf_gamma=tf.SRGB, gamut_name=cs.BT709)
    plot_result_pattern(
        result_img_fname="./img/test_on_ST2084_P3D65.png", div_num=6,
        tf_gamma=tf.ST2084, gamut_name=cs.P3_D65)

    # img = tpg.img_read_as_float("./img/test.png")
    # img = tf.eotf(img, tf.SRGB)
    # xyz = cs.rgb_to_large_xyz(img, cs.BT2020)
    # rgb = cs.large_xyz_to_rgb(img, cs.P3_D65)

    # img = np.clip(rgb, 0, 1) * 100
    # img = tf.oetf_from_luminance(img, tf.ST2084)
    # tpg.img_wirte_float_as_16bit_int("./img/test_on_ST2084_P3D65.png", img)
