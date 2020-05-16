# -*- coding: utf-8 -*-
"""
BT2407 実装用の各種LUTを作成する
===============================

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from multiprocessing import Pool, cpu_count, Array
import matplotlib.pyplot as plt
from colour.models import BT709_COLOURSPACE, BT2020_COLOURSPACE
from colour import Lab_to_XYZ, XYZ_to_RGB, RGB_to_XYZ, XYZ_to_Lab, Lab_to_LCHab

# import my libraries
import color_space as cs
import plot_utility as pu
from bt2407_parameters import L_SAMPLE_NUM_MAX, H_SAMPLE_NUM_MAX,\
    GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE, GAMUT_BOUNDARY_LUT_HUE_SAMPLE,\
    get_gamut_boundary_lut_name, get_l_cusp_name, get_focal_name,\
    get_chroma_map_lut_name
from bt2047_gamut_mapping import get_chroma_lightness_val_specfic_hue,\
    calc_chroma_lightness_using_length_from_l_focal,\
    calc_chroma_lightness_using_length_from_c_focal, calc_cusp_lut,\
    calc_degree_from_cl_data_using_c_focal,\
    calc_degree_from_cl_data_using_l_focal,\
    calc_distance_from_c_focal, calc_distance_from_l_focal,\
    eliminate_inner_gamut_data_c_focal, eliminate_inner_gamut_data_l_focal,\
    interpolate_chroma_map_lut, merge_lightness_mapping
from make_bt2047_luts import calc_value_from_hue_1dlut,\
    calc_chroma_map_degree2, calc_l_cusp_specific_hue, calc_cusp_in_lc_plane,\
    _calc_ab_coef_from_cl_point, solve_equation_for_intersection

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def lch_to_lab(lch):
    shape_bak = lch.shape
    aa = lch[..., 1] * np.cos(lch[..., 2])
    bb = lch[..., 1] * np.sin(lch[..., 2])
    return np.dstack((lch[..., 0], aa, bb)).reshape(shape_bak)


def print_blog_param_sub(
        rgb_2020=np.array([1023, 0, 0]), text="angle_40"):
    rgb_2020_linear = (rgb_2020 / 1023) ** 2.4
    lab_2020 = XYZ_to_Lab(
        RGB_to_XYZ(
            rgb_2020_linear, cs.D65, cs.D65,
            BT2020_COLOURSPACE.RGB_to_XYZ_matrix))
    lch_2020 = Lab_to_LCHab(lab_2020)

    print(f"rgb_2020_{text}={rgb_2020}")
    print(f"lab_2020_{text}={lab_2020}")
    print(f"lch_2020_{text}={lch_2020}")


def print_blog_param():
    """
    ブログ記載用のパラメータを吐く
    """
    rgb_40_2020 = np.array([1001, 509, 321])
    rgb_270_2020 = np.array([158, 421, 759])

    print_blog_param_sub(rgb_40_2020, "40")
    print_blog_param_sub(rgb_270_2020, "270")


def _make_debug_luminance_chroma_data_fixed_hue(cl_outer):
    dst_step = 64
    degree = np.linspace(-np.pi/2, np.pi/2, dst_step)
    a1 = np.tan(degree)
    b1 = 50 * np.ones_like(a1)
    a2, b2 = _calc_ab_coef_from_cl_point(cl_outer)
    out_chroma, out_lightness = solve_equation_for_intersection(
        cl_outer, a1, b1, a2, b2, focal="L_Focal")
    # chroma = cl_outer[..., 0]
    # lightness = cl_outer[..., 1]
    # step = GAMUT_BOUNDARY_LUT_HUE_SAMPLE // dst_step

    # out_chroma = np.append(chroma[::step], chroma[-1])
    # out_lightness = np.append(lightness[::step], lightness[-1])

    return out_lightness, out_chroma


def _check_chroma_map_lut_interpolation(
        hue_idx, hue,
        outer_color_space_name=cs.BT2020,
        inner_color_space_name=cs.BT709):
    """
    interpolate_chroma_map_lut() の動作確認用のデバッグコード。
    1. まずはLUT上の LC平面で確認
    2. 次に補間が働く LC平面で確認
    3. 今度は補間が働く ab平面で確認
    """
    print(hue_idx, np.rad2deg(hue))
    # とりあえず L*C* 平面のポリゴン準備
    cl_inner = get_chroma_lightness_val_specfic_hue(
        hue, get_gamut_boundary_lut_name(inner_color_space_name))
    cl_outer = get_chroma_lightness_val_specfic_hue(
        hue, get_gamut_boundary_lut_name(outer_color_space_name))

    # cusp 準備
    lh_inner_lut = np.load(
        get_gamut_boundary_lut_name(inner_color_space_name))
    inner_cusp = calc_cusp_in_lc_plane(hue, lh_inner_lut)
    lh_outer_lut = np.load(
        get_gamut_boundary_lut_name(outer_color_space_name))
    outer_cusp = calc_cusp_in_lc_plane(hue, lh_outer_lut)

    # l_cusp, l_focal, c_focal 準備
    l_cusp_lut = np.load(
        get_l_cusp_name(
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name))
    l_focal_lut = np.load(
        get_focal_name(
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name,
            focal_type="Lfocal"))
    c_focal_lut = np.load(
        get_focal_name(
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name,
            focal_type="Cfocal"))
    l_cusp = calc_value_from_hue_1dlut(hue, l_cusp_lut)
    l_focal = calc_value_from_hue_1dlut(hue, l_focal_lut)
    c_focal = calc_value_from_hue_1dlut(hue, c_focal_lut)

    # Chroma Mapping の Focalからの距離の LUT データ
    cmap_lut_c = np.load(
        get_chroma_map_lut_name(
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name,
            focal_type="Cfocal"))
    cmap_lut_l = np.load(
        get_chroma_map_lut_name(
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name,
            focal_type="Lfocal"))

    # st_degree, ed_degree を 1次元LUTの形で得る
    # st_degree_l[hue] = 30°, ed_degree_l[hue] = 120° 的な？
    inner_cusp_l_lut = calc_cusp_lut(lh_lut=lh_inner_lut)
    st_degree_l, ed_degree_l, st_degree_c, ed_degree_c =\
        calc_chroma_map_degree2(l_focal_lut, c_focal_lut, inner_cusp_l_lut)

    # とりあえず検証用のデータを準備
    # 一応、本番を想定して chroma-lightness から変換するように仕込む
    # hue-degree --> chroma-lightness --> hue_degree --> 補間的な？
    """ L_focal 基準データ """
    lightness_l, chroma_l = _make_debug_luminance_chroma_data_fixed_hue(
        cl_outer)
    hue_array = np.ones(chroma_l.shape[0]) * hue
    cl_data_l = np.dstack((chroma_l, lightness_l))[0]
    test_degree_l = calc_degree_from_cl_data_using_l_focal(
        cl_data=cl_data_l,
        l_focal=calc_value_from_hue_1dlut(hue_array, l_focal_lut))
    hd_data_l = np.dstack((hue_array, test_degree_l))[0]
    len_from_l_focal = calc_distance_from_l_focal(
        chroma_l, lightness_l, l_focal)

    """ C_focal 基準データ """
    lightness_c, chroma_c = _make_debug_luminance_chroma_data_fixed_hue(
        cl_outer)
    hue_array = np.ones(chroma_l.shape[0]) * hue
    cl_data_c = np.dstack((chroma_c, lightness_c))[0]
    test_degree_c = calc_degree_from_cl_data_using_c_focal(
        cl_data=cl_data_c,
        c_focal=calc_value_from_hue_1dlut(hue_array, c_focal_lut))
    hd_data_c = np.dstack((hue_array, test_degree_c))[0]
    len_from_c_focal = calc_distance_from_c_focal(
        chroma_c, lightness_c, c_focal)

    # まずは cmap_lut 値の Bilinear補間
    cmap_value_l = interpolate_chroma_map_lut(
        cmap_hd_lut=cmap_lut_l, degree_min=st_degree_l,
        degree_max=ed_degree_l, data_hd=hd_data_l)
    cmap_value_c = interpolate_chroma_map_lut(
        cmap_hd_lut=cmap_lut_c, degree_min=st_degree_c,
        degree_max=ed_degree_c, data_hd=hd_data_c)

    # 除外データは除外
    restore_idx_l = (len_from_l_focal <= cmap_value_l)
    cmap_value_l[restore_idx_l] = len_from_l_focal[restore_idx_l]
    restore_idx_c = (len_from_c_focal > cmap_value_c)
    cmap_value_c[restore_idx_c] = len_from_c_focal[restore_idx_c]

    # 補間して得られた cmap 値から CL平面上における座標を取得
    icn_x_l, icn_y_l = calc_chroma_lightness_using_length_from_l_focal(
        distance=cmap_value_l, degree=test_degree_l, l_focal=l_focal)
    icn_x_c, icn_y_c = calc_chroma_lightness_using_length_from_c_focal(
        distance=cmap_value_c, degree=test_degree_c, c_focal=c_focal)

    _debug_plot_check_lightness_mapping_specific_hue(
        hue, cl_inner, cl_outer, l_cusp, inner_cusp, outer_cusp,
        l_cusp, l_focal, c_focal,
        x_val=chroma_l, y_val=lightness_l, map_x=icn_x_l, map_y=icn_y_l,
        focal_type="L_focal", h_idx=hue_idx,
        outer_color_space_name=outer_color_space_name,
        inner_color_space_name=inner_color_space_name)
    _debug_plot_check_lightness_mapping_specific_hue(
        hue, cl_inner, cl_outer, l_cusp, inner_cusp, outer_cusp,
        l_cusp, l_focal, c_focal,
        x_val=chroma_c, y_val=lightness_c, map_x=icn_x_c, map_y=icn_y_c,
        focal_type="C_focal", h_idx=hue_idx,
        outer_color_space_name=outer_color_space_name,
        inner_color_space_name=inner_color_space_name)


def _check_lightness_mapping(
        hue_idx, hue,
        outer_color_space_name=cs.BT2020,
        inner_color_space_name=cs.BT709):
    """
    interpolate_chroma_map_lut() の動作確認用のデバッグコード。
    """
    print(hue_idx, np.rad2deg(hue))
    # とりあえず L*C* 平面のポリゴン準備
    cl_inner = get_chroma_lightness_val_specfic_hue(
        hue, get_gamut_boundary_lut_name(inner_color_space_name))
    cl_outer = get_chroma_lightness_val_specfic_hue(
        hue, get_gamut_boundary_lut_name(outer_color_space_name))

    # cusp 準備
    lh_inner_lut = np.load(
        get_gamut_boundary_lut_name(inner_color_space_name))
    inner_cusp = calc_cusp_in_lc_plane(hue, lh_inner_lut)
    lh_outer_lut = np.load(
        get_gamut_boundary_lut_name(outer_color_space_name))
    outer_cusp = calc_cusp_in_lc_plane(hue, lh_outer_lut)

    # l_cusp, l_focal, c_focal 準備
    l_cusp_lut = np.load(
        get_l_cusp_name(
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name))
    l_focal_lut = np.load(
        get_focal_name(
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name,
            focal_type="Lfocal"))
    c_focal_lut = np.load(
        get_focal_name(
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name,
            focal_type="Cfocal"))
    l_cusp = calc_value_from_hue_1dlut(hue, l_cusp_lut)
    l_focal = calc_value_from_hue_1dlut(hue, l_focal_lut)
    c_focal = calc_value_from_hue_1dlut(hue, c_focal_lut)

    # Chroma Mapping の Focalからの距離の LUT データ
    cmap_lut_c = np.load(
        get_chroma_map_lut_name(
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name,
            focal_type="Cfocal"))
    cmap_lut_l = np.load(
        get_chroma_map_lut_name(
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name,
            focal_type="Lfocal"))

    # st_degree, ed_degree を 1次元LUTの形で得る
    # st_degree_l[hue] = 30°, ed_degree_l[hue] = 120° 的な？
    inner_cusp_l_lut = calc_cusp_lut(lh_lut=lh_inner_lut)
    st_degree_l, ed_degree_l, st_degree_c, ed_degree_c =\
        calc_chroma_map_degree2(l_focal_lut, c_focal_lut, inner_cusp_l_lut)

    # とりあえず検証用のデータを準備
    # 一応、本番を想定して chroma-lightness から変換するように仕込む
    # hue-degree --> chroma-lightness --> hue_degree --> 補間的な？
    """ L_focal 基準データ """
    lightness_l, chroma_l = _make_debug_luminance_chroma_data_fixed_hue(
        cl_outer)
    hue_array = np.ones(chroma_l.shape[0]) * hue
    cl_data_l = np.dstack((chroma_l, lightness_l))[0]
    test_degree_l = calc_degree_from_cl_data_using_l_focal(
        cl_data=cl_data_l,
        l_focal=calc_value_from_hue_1dlut(hue_array, l_focal_lut))
    hd_data_l = np.dstack((hue_array, test_degree_l))[0]

    """ C_focal 基準データ """
    lightness_c, chroma_c = _make_debug_luminance_chroma_data_fixed_hue(
        cl_outer)
    hue_array = np.ones(chroma_l.shape[0]) * hue
    cl_data_c = np.dstack((chroma_c, lightness_c))[0]
    test_degree_c = calc_degree_from_cl_data_using_c_focal(
        cl_data=cl_data_c,
        c_focal=calc_value_from_hue_1dlut(hue_array, c_focal_lut))
    hd_data_c = np.dstack((hue_array, test_degree_c))[0]

    # まずは cmap_lut 値の Bilinear補間
    cmap_value_l = interpolate_chroma_map_lut(
        cmap_hd_lut=cmap_lut_l, degree_min=st_degree_l,
        degree_max=ed_degree_l, data_hd=hd_data_l)
    cmap_value_c = interpolate_chroma_map_lut(
        cmap_hd_lut=cmap_lut_c, degree_min=st_degree_c,
        degree_max=ed_degree_c, data_hd=hd_data_c)

    # out of gamut ではないデータは処理をしないようにする
    eliminate_inner_gamut_data_l_focal(
        dst_distance=cmap_value_l, src_chroma=chroma_l,
        src_lightness=lightness_l, l_focal=l_focal)
    eliminate_inner_gamut_data_c_focal(
        dst_distance=cmap_value_c, src_chroma=chroma_c,
        src_lightness=lightness_c, c_focal=c_focal)

    # 補間して得られた cmap 値から CL平面上における座標を取得
    icn_x_l, icn_y_l = calc_chroma_lightness_using_length_from_l_focal(
        distance=cmap_value_l, degree=test_degree_l, l_focal=l_focal)
    icn_x_c, icn_y_c = calc_chroma_lightness_using_length_from_c_focal(
        distance=cmap_value_c, degree=test_degree_c, c_focal=c_focal)

    # L_Focalベースと C_Focalベースの結果を統合
    icn_x, icn_y = merge_lightness_mapping(
        hd_data_l=hd_data_l, st_degree_l=st_degree_l,
        chroma_map_l=icn_x_l, lightness_map_l=icn_y_l,
        chroma_map_c=icn_x_c, lightness_map_c=icn_y_c)

    _debug_plot_check_lightness_mapping_specific_hue(
        hue, cl_inner, cl_outer, l_cusp, inner_cusp, outer_cusp,
        l_cusp, l_focal, c_focal,
        x_val=chroma_l, y_val=lightness_l, map_x=icn_x, map_y=icn_y,
        focal_type="All", h_idx=hue_idx,
        outer_color_space_name=outer_color_space_name,
        inner_color_space_name=inner_color_space_name)


def _debug_plot_check_lightness_mapping_specific_hue(
        hue, cl_inner, cl_outer, lcusp, inner_cusp, outer_cusp,
        l_cusp, l_focal, c_focal, x_val, y_val, map_x, map_y,
        focal_type, h_idx=0, outer_color_space_name=cs.BT2020,
        inner_color_space_name=cs.BT709):
    graph_title = f"HUE = {hue/2/np.pi*360:.1f}°, for {focal_type}"
    graph_title += f"={c_focal:.1f}" if focal_type == "C_focal" else ""
    fig1, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(16 * 0.9, 9 * 0.9),
        graph_title=graph_title,
        xlabel="Chroma",
        ylabel="Lightness",
        legend_size=17,
        xlim=[-10, 230],
        ylim=[-3, 103],
        xtick=[x * 20 for x in range(12)],
        ytick=[x * 10 for x in range(11)],
        return_figure=True)
    ax1.patch.set_facecolor("#E0E0E0")
    in_color = pu.BLUE
    ou_color = pu.RED
    fo_color = "#A0A0A0"
    src_color = pu.GREEN
    dst_color = pu.PINK

    # gamut boundary
    ax1.plot(
        cl_inner[..., 0], cl_inner[..., 1], c=in_color, label="BT.709")
    ax1.plot(cl_outer[..., 0], cl_outer[..., 1], c=ou_color, label="BT.2020")

    # gamut cusp
    ax1.plot(inner_cusp[1], inner_cusp[0], 's', ms=10, mec='k',
             c=in_color, label="BT.709 Cusp")
    ax1.plot(outer_cusp[1], outer_cusp[0], 's', ms=10, mec='k',
             c=ou_color, label="BT.2020 Cusp")

    # l_cusp, l_focal, c_focal
    ax1.plot([0], [l_cusp], 'x', ms=12, mew=4, c=in_color, label="L_cusp")
    ax1.plot([0], [l_focal], 'x', ms=12, mew=4, c=ou_color, label="L_focal")
    ax1.plot([c_focal], [0], '*', ms=12, mew=3, c=ou_color, label="C_focal")
    ax1.plot([0, c_focal], [l_focal, 0], '--', c='k')

    # intersectionx
    ax1.plot(x_val, y_val, 'o', ms=9, c=src_color, label="src point")
    ax1.plot(map_x, map_y, 'o', ms=6, c=dst_color, label="dst point")
    for x, y in zip(x_val, y_val):
        if y >= (-l_focal * x / c_focal + l_focal):
            aa = (y - l_focal) / x
            bb = l_focal
            xx = 230
            yy = aa * xx + bb
            ax1.plot([0, xx], [l_focal, yy], ':', c=fo_color)
        else:
            aa = (y) / (x - c_focal)
            bb = y - aa * x
            xx = 0
            yy = aa * xx + bb
            ax1.plot([0, c_focal], [yy, 0], ':', c=fo_color)

    # annotation
    diff = ((map_x - x_val) ** 2 + (map_y - y_val) ** 2) ** 0.5
    arrowprops = dict(
        facecolor='#333333', shrink=0.0, headwidth=4, headlength=5,
        width=1)
    for idx in range(len(map_x)):
        if diff[idx] > 0.01:
            st_pos = (x_val[idx], y_val[idx])
            ed_pos = (map_x[idx], map_y[idx])
            ax1.annotate(
                "", xy=ed_pos, xytext=st_pos, xycoords='data',
                textcoords='data', ha='left', va='bottom',
                arrowprops=arrowprops)

    graph_name = f"./video_src/lightness_mapping_"\
        + f"{outer_color_space_name}_to_{inner_color_space_name}_"\
        + f"{focal_type}_{h_idx:04d}.png"
    plt.legend(loc='upper right')
    print(graph_name)
    # plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(graph_name)  # オプション付けるとエラーになるので外した
    # plt.show()
    plt.close(fig1)


def _check_upper_and_lower_mapping(
        hue_sample_num=10,
        outer_color_space_name=cs.BT2020,
        inner_color_space_name=cs.BT709):
    hue_list = np.deg2rad(
        np.linspace(0, 360, hue_sample_num, endpoint=False))
    args = []
    for idx, hue in enumerate(hue_list):
        # _check_chroma_map_lut_interpolation(
        #     hue_idx=idx, hue=hue,
        #     outer_color_space_name=cs.BT2020,
        #     inner_color_space_name=cs.BT709)
        d = dict(
            hue_idx=idx, hue=hue,
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name)
        args.append(d)
    with Pool(cpu_count()) as pool:
        pool.map(thread_wrapper_check_chroma_map_lut_interpolation, args)


def thread_wrapper_check_chroma_map_lut_interpolation(args):
    _check_chroma_map_lut_interpolation(**args)


def _check_lightness_mapping_seq(
        hue_sample_num=16,
        outer_color_space_name=cs.BT2020,
        inner_color_space_name=cs.BT709):
    hue_list = np.deg2rad(
        np.linspace(0, 360, hue_sample_num, endpoint=False))
    args = []
    for idx, hue in enumerate(hue_list):
        # _check_lightness_mapping(
        #     hue_idx=idx, hue=hue,
        #     outer_color_space_name=cs.BT2020,
        #     inner_color_space_name=cs.BT709)
        d = dict(
            hue_idx=idx, hue=hue,
            outer_color_space_name=outer_color_space_name,
            inner_color_space_name=inner_color_space_name)
        args.append(d)
    with Pool(cpu_count()) as pool:
        pool.map(thread_wrapper_check_lightness_mapping, args)


def thread_wrapper_check_lightness_mapping(args):
    _check_lightness_mapping(**args)


def main_func():
    # print_blog_param()
    # _check_chroma_map_lut_interpolation(0, np.deg2rad(30))
    # _check_upper_and_lower_mapping(
    #     hue_sample_num=1025,
    #     outer_color_space_name=cs.BT2020,
    #     inner_color_space_name=cs.BT709)
    # _check_upper_and_lower_mapping(
    #     hue_sample_num=1025,
    #     outer_color_space_name=cs.P3_D65,
    #     inner_color_space_name=cs.BT709)
    _check_lightness_mapping_seq(
        hue_sample_num=16,
        outer_color_space_name=cs.BT2020,
        inner_color_space_name=cs.BT709)
    _check_lightness_mapping_seq(
        hue_sample_num=16,
        outer_color_space_name=cs.P3_D65,
        inner_color_space_name=cs.BT709)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
