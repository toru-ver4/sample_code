# -*- coding: utf-8 -*-
"""

==========

"""

# import standard libraries
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

# import third-party libraries
from colour.utilities import tstack
from colour import XYZ_to_xy, XYZ_to_xyY, xyY_to_XYZ
import numpy as np
from numpy import linalg

# import my libraries
import test_pattern_generator2 as tpg
from test_pattern_coordinate import GridCoordinate
import transfer_functions as tf
import font_control2 as fc2
import color_space as cs
import plot_utility as pu
from ty_utility import add_suffix_to_filename
from ty_algebra import calc_y_from_three_pos
from create_gamut_booundary_lut import create_Ych_gamut_boundary_lut,\
    make_Ych_gb_lut_fname, TyLchLut, calc_l_focal_specific_hue_jzazbz

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
        stating point. [x1, y1, Y1]
    pos2 : list
        ending point. [x2, y2, Y1]
    n : int
        division number

    Returns
    -------
    ndarray
        position list

    Examples
    --------
    >>> pos1 = np.array([0.5, 0.5, 0.5])
    >>> pos2 = np.array([1.0, 0.7, 0.8])
    >>> pos_list = cut_two_points_into_N_points(pos1, pos2, 3)
    >>> print(pos_list)
    [[ 0.5   0.5   0.5 ]
     [ 0.75  0.6   0.65]
     [ 1.    0.7   0.8 ]
    """
    x1, y1, Y1 = pos1
    x2, y2, Y2 = pos2
    x = np.linspace(x1, x2, n)
    y = np.linspace(y1, y2, n)
    Y = np.linspace(Y1, Y2, n)
    pos_list = tstack([x, y, Y])

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


def calc_n_devided_Ych_point(Ych, div_num):
    hue_sample = 1001
    lightness_sample = 1001
    h_val = Ych[2]
    ll_base = np.linspace(0, 1, lightness_sample)
    hh_base = np.ones_like(ll_base) * h_val
    lh_array = tstack([ll_base, hh_base])
    lut_bt2020_name = make_Ych_gb_lut_fname(
        color_space_name=cs.BT2020, lightness_num=lightness_sample,
        hue_num=hue_sample)
    lut_bt709_name = make_Ych_gb_lut_fname(
        color_space_name=cs.BT709, lightness_num=lightness_sample,
        hue_num=hue_sample)
    lut_bt2020 = TyLchLut(np.load(lut_bt2020_name))
    lut_bt709 = TyLchLut(np.load(lut_bt709_name))

    ych_2020 = lut_bt2020.interpolate(lh_array=lh_array)
    ych_709 = lut_bt709.interpolate(lh_array=lh_array)
    rgb_2020 = ych_to_rgb_srgb(ych_2020, cs_name=cs.BT2020)
    rgb_709 = ych_to_rgb_srgb(ych_709, cs_name=cs.BT2020)

    cusp_2020 = lut_bt2020.get_cusp(hue=h_val, ych=True)
    cusp_709 = lut_bt709.get_cusp(hue=h_val, ych=True)

    pos1 = [0, 0.20]
    pos2 = [cusp_709[1], cusp_709[0]]
    pos3 = [cusp_2020[1], cusp_2020[0]]

    chroma = np.linspace(pos1[0], pos3[0], div_num)
    hue = np.ones_like(chroma) * h_val
    yy = calc_y_from_three_pos(chroma, pos1, pos2, pos3)

    Ych_list = tstack([yy, chroma, hue])

    return Ych_list


def calc_6color_xyY_coordinate(div_num=6):
    """
    Returns
    -------
    ndarray
        rgb list. shape is (num_of_color, div_num, 3).

    Examples
    --------
    >>> calc_6color_xyY_coordinate(div_num=3)
    [[[ 0.3127      0.329       0.5       ]
      [ 0.51035     0.3105      0.38135011]
      [ 0.708       0.292       0.26270021]]

     [[ 0.3127      0.329       0.5       ]
      [ 0.379613    0.43321782  0.72034914]
      [ 0.44652599  0.53743564  0.94069828]]

     [[ 0.3127      0.329       0.5       ]
      [ 0.24135     0.563       0.58899904]
      [ 0.17        0.797       0.67799807]]

     [[ 0.3127      0.329       0.5       ]
      [ 0.22960212  0.33677795  0.61864989]
      [ 0.14650423  0.34455589  0.73729979]]

     [[ 0.3127      0.329       0.5       ]
      [ 0.22185     0.1875      0.27965086]
      [ 0.131       0.046       0.05930172]]

     [[ 0.3127      0.329       0.5       ]
      [ 0.34043019  0.23805585  0.41100096]
      [ 0.36816038  0.14711171  0.32200193]]]
    """
    # white = np.array([0.3127, 0.3290, 0.18])
    rygcbm_rgb = np.array([
        [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1],
        [0, 0, 1], [1, 0, 1]])
    large_xyz = cs.rgb_to_large_xyz(rygcbm_rgb, cs.BT2020)
    xyY = XYZ_to_xyY(large_xyz)
    Ych = cs.xyY_to_Ych(xyY)
    num_of_color = len(xyY)
    xyY_list = np.zeros((num_of_color, div_num, 3))

    for idx, ed_pos in enumerate(xyY):
        Ych_one_color = calc_n_devided_Ych_point(
            Ych=Ych[idx], div_num=div_num)
        xyY_list[idx] = cs.Ych_to_xyY(Ych_one_color)
        # xy_one_color = cut_two_points_into_N_points(white, ed_pos, div_num)
        # xyY_list[idx] = xy_one_color

    return xyY_list


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
    xyY = calc_6color_xyY_coordinate(div_num=div_num)
    large_xyz = xyY_to_XYZ(xyY)
    rgb = cs.large_xyz_to_rgb(large_xyz, cs.BT2020)
    # max_list = np.max(rgb, axis=-1)[:, :, np.newaxis]
    rgb = np.clip(rgb, 0, 1)

    return rgb


def make_patch_center_pos_list_fname(div_num):
    return f"./debug/patch_center_pos_div_num-{div_num}.npy"


def draw_patch(img, div_num=6):
    height, width = img.shape[:2]
    rgb_list = calc_6color_rgb_val(div_num=div_num)
    # print(rgb_list)
    num_of_color, div_num = rgb_list.shape[:2]
    size = 128

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


def make_srgb_bt2020_tp_fname(width, height, revision):
    return f"./img/wcg_tp_{width}x{height}_rev{revision}.png"


def create_gamut_evaluation_pattern(div_num=7):
    width = 1920
    total_height = 1080
    info_area_height_rate = 0.04
    revision = 1
    text1 = "WCG Pattern, EOTF: sRGB, Gamut: BT.2020, "
    text1 += f"White point: D65, {width}x{total_height}"
    text2 = f"Rev.{revision}"
    info_img = carete_tp_bottom_information_image(
        width=width, height=total_height,
        text1=text1, text2=text2,
        height_rate=info_area_height_rate)
    height = total_height - info_img.shape[0]
    img = np.ones((height, width, 3)) * np.array([0.015, 0.015, 0.015])
    draw_patch(img=img, div_num=div_num)

    img = np.vstack((img, info_img))

    img = tf.oetf(img, tf.SRGB)
    icc_profile_name = "../../2020/026_create_icc_profiles/sRGB_BT2020.icc"
    img_fname = make_srgb_bt2020_tp_fname(width, total_height, revision)
    print(img_fname)
    tpg.img_wirte_float_as_16bit_int_with_icc(
        filename=img_fname, img_float=img,
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
    # print(result_rgb)
    result_xyY = XYZ_to_xyY(cs.rgb_to_large_xyz(result_rgb, gamut_name))
    ref_xyY = calc_6color_xyY_coordinate(div_num=div_num)
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
        ref_xyY[..., 0], ref_xyY[..., 1],
        s=ref_ms*rate, c=ref_rgb.reshape(-1, 3), zorder=10,
        edgecolors='k', linewidth=1.5*rate)
    ax1.scatter(
        result_xyY[..., 0], result_xyY[..., 1], marker="X",
        s=measure_ms*rate, c=ref_rgb.reshape(-1, 3), zorder=10,
        linewidth=1.5*rate, edgecolors='k')

    basename = Path(result_img_fname).stem
    plot_fname = f"./graph/result_{basename}.png"
    print(plot_fname)
    pu.show_and_save(
        fig=fig, legend_loc='upper right', save_fname=plot_fname)


def conv_sRGB_BT2020_to_ST2084_P3D65():
    src_img_fname = make_srgb_bt2020_tp_fname(
        width=1920, height=1080, revision=1)
    img = tpg.img_read_as_float(src_img_fname)
    img = tf.eotf(img, tf.SRGB)
    xyz = cs.rgb_to_large_xyz(img, cs.BT2020)
    rgb = cs.large_xyz_to_rgb(xyz, cs.P3_D65)
    img = np.clip(rgb, 0, 1) * 100
    img = tf.oetf_from_luminance(img, tf.ST2084)
    dst_img_fname = add_suffix_to_filename(
        src_img_fname, suffix="_on_ST2084_P3D65")
    print(dst_img_fname)
    tpg.img_wirte_float_as_16bit_int(dst_img_fname, img)


def conv_sRGB_BT2020_to_ST2084_BT709():
    src_img_fname = make_srgb_bt2020_tp_fname(
        width=1920, height=1080, revision=1)
    img = tpg.img_read_as_float(src_img_fname)
    img = tf.eotf(img, tf.SRGB)
    xyz = cs.rgb_to_large_xyz(img, cs.BT2020)
    rgb = cs.large_xyz_to_rgb(xyz, cs.BT709)
    img = np.clip(rgb, 0, 1) * 100
    img = tf.oetf_from_luminance(img, tf.ST2084)
    dst_img_fname = add_suffix_to_filename(
        src_img_fname, suffix="_on_ST2084_BT709")
    print(dst_img_fname)
    tpg.img_wirte_float_as_16bit_int(dst_img_fname, img)


def ych_to_rgb_srgb(ych, cs_name=cs.BT2020):
    xyY = cs.Ych_to_xyY(ych)
    large_xyz = xyY_to_XYZ(xyY)
    rgb = cs.large_xyz_to_rgb(xyz=large_xyz, color_space_name=cs_name)
    rgb_srgb = tf.oetf(np.clip(rgb, 0.0, 1.0), tf.SRGB)

    return rgb_srgb


def solve_quadratic_func_param(pt1, pt2, pt3):
    x1, y1 = pt1
    x2, y2 = pt2
    x3, y3 = pt3

    left_equation = np.array(
        [[x1 ** 2, x1, 1], [x2 ** 2, x2, 1], [x3 ** 2, x3, 1]])
    right_equation = np.array([y1, y2, y3])

    result_data = linalg.solve(left_equation, right_equation)

    return result_data


def thread_wrapper_plot_YCH_plane(args):
    plot_YCH_plane(**args)


def plot_YCH_plane(h_idx=0, h_val=0):
    hue_sample = 1001
    lightness_sample = 1001
    ll_base = np.linspace(0, 1, lightness_sample)
    hh_base = np.ones_like(ll_base) * h_val
    lh_array = tstack([ll_base, hh_base])
    lut_bt2020_name = make_Ych_gb_lut_fname(
        color_space_name=cs.BT2020, lightness_num=lightness_sample,
        hue_num=hue_sample)
    lut_bt709_name = make_Ych_gb_lut_fname(
        color_space_name=cs.BT709, lightness_num=lightness_sample,
        hue_num=hue_sample)
    lut_bt2020 = TyLchLut(np.load(lut_bt2020_name))
    lut_bt709 = TyLchLut(np.load(lut_bt709_name))

    ych_2020 = lut_bt2020.interpolate(lh_array=lh_array)
    ych_709 = lut_bt709.interpolate(lh_array=lh_array)
    rgb_2020 = ych_to_rgb_srgb(ych_2020, cs_name=cs.BT2020)
    rgb_709 = ych_to_rgb_srgb(ych_709, cs_name=cs.BT2020)

    cusp_2020 = lut_bt2020.get_cusp(hue=h_val, ych=True)
    cusp_709 = lut_bt709.get_cusp(hue=h_val, ych=True)

    pos1 = [0, 0.20]
    pos2 = [cusp_709[1], cusp_709[0]]
    pos3 = [cusp_2020[1], cusp_2020[0]]

    x = np.linspace(pos1[0], pos3[0], 32)
    y = calc_y_from_three_pos(x, pos1, pos2, pos3)

    cc_2020 = ych_2020[..., 1]
    cc_709 = ych_709[..., 1]
    ll_2020 = ych_2020[..., 0]
    ll_709 = ych_709[..., 0]

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 12),
        bg_color=(0.5, 0.5, 0.5),
        graph_title=f"YCH, hue-angle={h_val:.02f}°",
        graph_title_size=None,
        xlabel="Chroma", ylabel="Y",
        axis_label_size=None,
        legend_size=17,
        xlim=[0, 0.5],
        ylim=[0.0009, 1.0],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=1,
        minor_xtick_num=None,
        minor_ytick_num=None)
    # pu.log_sacle_settings_x_linear_y_log(ax=ax1)
    ax1.scatter(cc_2020, ll_2020, c=rgb_2020.reshape(-1, 3))
    ax1.scatter(cc_709, ll_709, c=rgb_709.reshape(-1, 3))
    ax1.plot(x, y, 'o', color='k', ms=10)
    fname = "/work/overuse/2022/09_create_mhc_icc_profile/ych/"
    fname += f"lch_{h_idx:04d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)


def plot_ych_ch_plane_all_hue():
    hue_sample = 721
    hue_list = np.linspace(0, 360, hue_sample)

    total_frame = hue_sample
    total_process_num = total_frame
    block_process_num = int(cpu_count() * 0.8)
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            l_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={l_idx}")  # User
            if l_idx >= total_process_num:                         # User
                break
            d = dict(h_idx=l_idx, h_val=hue_list[l_idx])
            args.append(d)
        #     plot_YCH_plane(**d)
        #     break
        # break
        with Pool(block_process_num) as pool:
            pool.map(thread_wrapper_plot_YCH_plane, args)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    div_num = 7
    create_gamut_evaluation_pattern(div_num=div_num)
    conv_sRGB_BT2020_to_ST2084_P3D65()
    conv_sRGB_BT2020_to_ST2084_BT709()
    plot_result_pattern(
        result_img_fname="./img/wcg_tp_1920x1080_rev1.png", div_num=6,
        tf_gamma=tf.SRGB, gamut_name=cs.BT709)
    plot_result_pattern(
        result_img_fname="./img/wcg_tp_1920x1080_rev1_on_ST2084_P3D65.png",
        div_num=div_num, tf_gamma=tf.ST2084, gamut_name=cs.P3_D65)
    plot_result_pattern(
        result_img_fname="./img/wcg_tp_1920x1080_rev1_on_ST2084_BT709.png",
        div_num=div_num, tf_gamma=tf.ST2084, gamut_name=cs.BT709)

    # create_Ych_gamut_boundary_lut(
    #     hue_sample=1001, lightness_sample=1001, chroma_sample=16384,
    #     color_space_name=cs.BT2020)
    # create_Ych_gamut_boundary_lut(
    #     hue_sample=1001, lightness_sample=1001, chroma_sample=16384,
    #     color_space_name=cs.BT709)

    # plot_ych_ch_plane_all_hue()
