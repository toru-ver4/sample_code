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
from colour import RGB_to_XYZ, XYZ_to_Lab, Lab_to_XYZ, XYZ_to_RGB,\
    RGB_COLOURSPACES

# import my libraries
from bt2407_parameters import get_chroma_map_lut_name,\
    get_gamut_boundary_lut_name
import color_space as cs
from cielab import bilinear_interpolation

from bt2407_parameters import GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE,\
    GAMUT_BOUNDARY_LUT_HUE_SAMPLE, get_l_cusp_name, get_focal_name
from make_bt2047_luts import calc_value_from_hue_1dlut, load_cusp_focal_lut,\
    calc_chroma_map_degree2


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def merge_lightness_mapping(
        hd_data_l, st_degree_l,
        chroma_map_l, lightness_map_l, chroma_map_c, lightness_map_c):
    """
    L_Focalベース, C_Focalベースの結果をマージする。
    具体的には、入力の hd_data_l の degree に対して、
    L_Focal の開始 degree よりも大きい場合は L_Focal の結果を、
    それ意外は C_Focal の結果を使うようにしている。

    Parameters
    ----------
    hd_data_l : array_like
        L_focal ベースの hue-degree のデータ
    st_degree_l : array_like
        chroma mapping 用の hue-degree 2DLUT の各HUEに対する
        開始 degree の入ったデータ
    chroma_map_l : array_like
        L_Focal ベースで Lightness Mapping したあとの Chroma値
    lightness_map_l : array_like
        L_Focal ベースで Lightness Mapping したあとの Lightness値
    chroma_map_c : array_like
        C_Focal ベースで Lightness Mapping したあとの Chroma値
    lightness_map_c : array_like
        C_Focal ベースで Lightness Mapping したあとの Lightness値
    """
    # 出力用バッファ用意
    chroma_out = np.zeros_like(chroma_map_l)
    lightness_out = np.zeros_like(lightness_map_l)

    # 上側・下側のデータを後で抜き出すためのindexを計算
    st_degree_l_intp = calc_value_from_hue_1dlut(
        hd_data_l[..., 0], st_degree_l)
    upper_area_idx = (hd_data_l[..., 1] >= st_degree_l_intp)
    lower_area_idx = np.logical_not(upper_area_idx)

    # L_focal と C_focal の結果をマージ
    chroma_out[upper_area_idx] = chroma_map_l[upper_area_idx]
    lightness_out[upper_area_idx] = lightness_map_l[upper_area_idx]
    chroma_out[lower_area_idx] = chroma_map_c[lower_area_idx]
    lightness_out[lower_area_idx] = lightness_map_c[lower_area_idx]

    return chroma_out, lightness_out


def eliminate_inner_gamut_data_l_focal(
        dst_distance, src_chroma, src_lightness, l_focal):
    """
    元々の Gamut の範囲内のデータは Lightness Mapping を
    しないように元のデータに戻す。

    実は Lightness Mapping では Gamutの範囲内外もすべて
    Gamut の境界線上にマッピングしてしまっている（分岐を減らすため）。
    当然、Mapping が不要なデータは戻すべきであり、本関数ではその処理を行う。

    ここでは Luminance Mapping の前後での Focal からの distance を
    比較している。前述の通り、Luminance Mapping では Gamut の内外を問わず
    全て Gamut の境界線上にマッピングしている。したがって、
    `src_distance <= dst_distance` の配列のデータを元に戻せば良い。

    Parameters
    ----------
    dst_distance : array_like
        distance from L_focal after luminance mapping.
    src_chroma : array_like
        chroma value before luminance mapping.
    lightness : array_like
        lightness value before luminance mapping.
    """
    src_distance = calc_distance_from_l_focal(
        src_chroma, src_lightness, l_focal)
    restore_idx_l = (src_distance <= dst_distance)
    dst_distance[restore_idx_l] = src_distance[restore_idx_l]


def eliminate_inner_gamut_data_c_focal(
        dst_distance, src_chroma, src_lightness, c_focal):
    """
    元々の Gamut の範囲内のデータは Lightness Mapping を
    しないように元のデータに戻す。

    実は Lightness Mapping では Gamutの範囲内外もすべて
    Gamut の境界線上にマッピングしてしまっている（分岐を減らすため）。
    当然、Mapping が不要なデータは戻すべきであり、本関数ではその処理を行う。

    ここでは Luminance Mapping の前後での Focal からの distance を
    比較している。前述の通り、Luminance Mapping では Gamut の内外を問わず
    全て Gamut の境界線上にマッピングしている。したがって、
    `src_distance > dst_distance` の配列のデータを元に戻せば良い。

    Parameters
    ----------
    dst_distance : array_like
        distance from L_focal after luminance mapping.
    src_chroma : array_like
        chroma value before luminance mapping.
    lightness : array_like
        lightness value before luminance mapping.
    """
    src_distance = calc_distance_from_c_focal(
        src_chroma, src_lightness, c_focal)
    restore_idx_c = (src_distance > dst_distance)
    dst_distance[restore_idx_c] = src_distance[restore_idx_c]


def interpolate_chroma_map_lut(cmap_hd_lut, degree_min, degree_max, data_hd):
    """
    Chroma Mapping の LUT が任意の Input に
    対応できるように補間をする。

    LUTは Hue-Degree の2次元LUTである。
    これを Bilinear補間する。

    cmap_hd_lut: array_like
        Hue, Degree に対応する Chroma値が入っているLUT。

    degree_min: array_like
        cmap_hd_lut の 各 h_idx に対する degree の
        開始角度(degree_min)、終了角度(degree_max) が
        入っている。
        print(degree_min[h_idx]) ==> 0.4pi みたいなイメージ

    data_hd: array_like(shape is (N, 2))
        data_hd[..., 0]: Hue Value
        data_hd[..., 1]: Degree

    """
    # 補間に利用するLUTのIndexを算出
    hue_data = data_hd[..., 0]
    degree_data = data_hd[..., 1]
    hue_sample_num = GAMUT_BOUNDARY_LUT_HUE_SAMPLE
    hue_index_max = hue_sample_num - 1
    degree_sample_num = GAMUT_BOUNDARY_LUT_HUE_SAMPLE
    degree_index_max = degree_sample_num - 1

    # 1. h_idx
    h_idx_float = hue_data / (2 * np.pi) * (hue_index_max)
    h_idx_low = np.int16(h_idx_float)
    h_idx_high = h_idx_low + 1
    h_idx_low = np.clip(h_idx_low, 0, hue_index_max)
    h_idx_high = np.clip(h_idx_high, 0, hue_index_max)

    degree_lmin = degree_min[h_idx_low]
    degree_lmax = degree_max[h_idx_low]
    degree_hmin = degree_min[h_idx_high]
    degree_hmax = degree_max[h_idx_high]

    # 2. d_idx
    d_idx_l_float = (degree_data - degree_lmin)\
        / (degree_lmax - degree_lmin) * degree_index_max
    d_idx_l_float = np.clip(d_idx_l_float, 0, degree_index_max)

    d_idx_ll = np.int16(d_idx_l_float)
    d_idx_lh = d_idx_ll + 1
    d_idx_h_float = (degree_data - degree_hmin)\
        / (degree_hmax - degree_hmin) * degree_index_max
    d_idx_h_float = np.clip(d_idx_h_float, 0, degree_index_max)
    d_idx_hl = np.int16(d_idx_h_float)
    d_idx_hh = d_idx_hl + 1
    d_idx_ll = np.clip(d_idx_ll, 0, degree_index_max)
    d_idx_lh = np.clip(d_idx_lh, 0, degree_index_max)
    d_idx_hl = np.clip(d_idx_hl, 0, degree_index_max)
    d_idx_hh = np.clip(d_idx_hh, 0, degree_index_max)

    # 3. r_low, r_high
    r_low = d_idx_lh - d_idx_l_float
    r_high = d_idx_hh - d_idx_h_float

    # 4. interpolation in degree derection
    intp_d_low = r_low * cmap_hd_lut[h_idx_low, d_idx_ll]\
        + (1 - r_low) * cmap_hd_lut[h_idx_low, d_idx_lh]
    intp_d_high = r_high * cmap_hd_lut[h_idx_high, d_idx_hl]\
        + (1 - r_high) * cmap_hd_lut[h_idx_high, d_idx_hh]

    # 6. final_r
    final_r = h_idx_high - h_idx_float

    # 7. interpolation in hue direction
    intp_data = final_r * intp_d_low + (1 - final_r) * intp_d_high

    return intp_data


def calc_distance_from_l_focal(chroma, lightness, l_focal):
    """
    L_Focal から 引数で指定した Chroma-Lightness までの距離を求める。
    """
    distance = ((chroma) ** 2 + (lightness - l_focal) ** 2) ** 0.5
    return distance


def calc_distance_from_c_focal(chroma, lightness, c_focal):
    """
    C_Focal から 引数で指定した Chroma-Lightness までの距離を求める。
    """
    distance = ((chroma - c_focal) ** 2 + (lightness) ** 2) ** 0.5
    return distance


def calc_degree_from_cl_data_using_l_focal(cl_data, l_focal):
    """
    chroma-lightness のデータから degree を計算
    """
    chroma = cl_data[..., 0]
    lightness = cl_data[..., 1]

    # chroma == 0 は -np.pi/2 or np.pi/2 になる
    degree = np.where(
        chroma != 0,
        np.arctan((lightness - l_focal) / chroma),
        (np.pi / 2) * np.sign(lightness - l_focal)
    )

    return degree


def calc_degree_from_cl_data_using_c_focal(cl_data, c_focal):
    """
    chroma-lightness のデータから degree を計算
    """
    chroma = cl_data[..., 0]
    lightness = cl_data[..., 1]

    return np.arctan(lightness / (chroma - c_focal)) + np.pi


def calc_cusp_lut(lh_lut):
    """
    Gamut Boundary の Lightness-Hue の LUTから
    Cusp の (Lightness, Chroma) 情報が入った LUT を作る。

    Parameters
    ----------
    lh_lut : array_like
        Gamut Bondary の lh_lut[L_idx, H_idx] = Chroma 的なやつ。

    Returns
    -------
    array_like
        H_idx を入れると Lightness, Chroma が得られる LUT。
        retun_val[h_idx, 0] => Lightness
        retun_val[h_idx, 1] => Chroma 的な。
    """
    cusp_chroma = np.max(lh_lut, axis=0)
    cusp_chroma_idx = np.argmax(lh_lut, axis=0)
    cusp_lightness = cusp_chroma_idx / (lh_lut.shape[0] - 1) * 100

    return np.dstack((cusp_lightness, cusp_chroma))[0]


def get_chroma_lightness_val_specfic_hue(
        hue=30/360*2*np.pi,
        lh_lut_name=get_gamut_boundary_lut_name(cs.BT709)):
    """
    Gamut Boundary の LUT から 任意の HUE の Chroma-Lighenss を得る。

    以下のコマンドで 横軸 Chroma、縦軸 Lightness の平面が書ける。
    ```
    plt.plot(retun_vall[..., 0], retun_vall[..., 1])
    plt.show()
    ```
    """
    lh_lut = np.load(lh_lut_name)
    lstar = np.linspace(0, 100, lh_lut.shape[0])
    hue_list = np.ones((lh_lut.shape[1])) * hue
    lh = np.dstack([lstar, hue_list])
    chroma = bilinear_interpolation(lh, lh_lut)

    return np.dstack((chroma, lstar))[0]


def calc_chroma_lightness_using_length_from_l_focal(
        distance, degree, l_focal):
    """
    L_Focal からの距離(distance)から chroma, lightness 値を
    三角関数の計算で算出する。

    Parameters
    ----------
    distance : array_like
        Chroma-Lightness 平面における L_focal からの距離の配列。
        例えば Lightness Mapping 後の距離が入ってたりする。
    degree : array_like
        L_focal からの角度。Chroma軸が0°である。
    l_focal : array_like
        l_focal の配列。Lightness の値のリスト。

    Returns
    ------
    chroma : array_like
        Chroma値
    lightness : array_like
        Lightness値
    """
    chroma = distance * np.cos(degree)
    lightness = distance * np.sin(degree) + l_focal

    return chroma, lightness


def calc_chroma_lightness_using_length_from_c_focal(
        distance, degree, c_focal):
    """
    C_Focal からの距離(distance)から chroma, lightness 値を
    三角関数の計算で算出する。

    Parameters
    ----------
    distance : array_like
        Chroma-Lightness 平面における C_focal からの距離の配列。
        例えば Lightness Mapping 後の距離が入ってたりする。
    degree : array_like
        C_focal からの角度。Chroma軸が0°である。
    c_focal : array_like
        c_focal の配列。Chroma の値のリスト。

    Returns
    ------
    chroma : array_like
        Chroma値
    lightness : array_like
        Lightness値
    """
    chroma = distance * np.cos(degree) + c_focal
    lightness = distance * np.sin(degree)

    return chroma, lightness


def calc_hue_from_ab(aa, bb):
    """
    CIELAB空間で a, b の値から HUE を計算する。
    出力の値域は [0, 2pi) である。

    Examples
    --------
    >>> aa=np.array([1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0.0, 0.5, 0.99])*np.pi,
    >>> bb=np.array([0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5, -0.001])*np.pi
    >>> hue = calc_hue_from_ab(aa, bb)
    [0.  45.  90.  135.  180.  225.  270.  315. 359.94212549]
    """
    hue = np.where(aa != 0, np.arctan(bb/aa), np.pi/2*np.sign(bb))
    add_pi_idx = (aa < 0) & (bb >= 0)
    sub_pi_idx = (aa < 0) & (bb < 0)
    hue[add_pi_idx] = hue[add_pi_idx] + np.pi
    hue[sub_pi_idx] = hue[sub_pi_idx] - np.pi

    hue[hue < 0] = hue[hue < 0] + 2 * np.pi

    return hue


def calc_hue_degree_data_from_rgb(
        rgb_linear, l_focal_lut, c_focal_lut,
        outer_color_space_name=cs.BT2020):
    """
    Lightness Mapping に使用する Hue-Degree 形式に変換する。
    """
    large_xyz = RGB_to_XYZ(
        rgb_linear, cs.D65, cs.D65,
        RGB_COLOURSPACES[outer_color_space_name].RGB_to_XYZ_matrix)
    lab = XYZ_to_Lab(large_xyz)

    lightness = lab[..., 0]
    aa = lab[..., 1]
    bb = lab[..., 2]

    hue = calc_hue_from_ab(aa, bb)
    chroma = ((aa ** 2) + (bb ** 2)) ** 0.5
    # _debug_plot_ab_plane(rgb_linear, hue, chroma)

    cl_data = np.dstack((chroma, lightness))[0]

    degree_l = calc_degree_from_cl_data_using_l_focal(
        cl_data=cl_data,
        l_focal=calc_value_from_hue_1dlut(hue, l_focal_lut))
    degree_c = calc_degree_from_cl_data_using_c_focal(
        cl_data=cl_data,
        c_focal=calc_value_from_hue_1dlut(hue, c_focal_lut))

    hd_data_l = np.dstack((hue, degree_l))[0]
    hd_data_c = np.dstack((hue, degree_c))[0]

    return hd_data_l, hd_data_c, chroma, lightness


def luminance_mapping_in_hd_space(
        hd_data_l, hd_data_c, src_chroma, src_lightness,
        l_focal_lut, c_focal_lut,
        outer_color_space_name=cs.BT2020,
        inner_color_space_name=cs.BT709):
    """
    BT2407 の Gamut Mapping のコア処理部分。
    Hue-Degree 空間での Luminance Mapping の実行
    """

    hue = hd_data_l[..., 0]
    l_focal = calc_value_from_hue_1dlut(hue, l_focal_lut)
    c_focal = calc_value_from_hue_1dlut(hue, c_focal_lut)
    """
    cmap_lut には Mapping 先の座標情報として
    各Focal からの distance が記録されている。
    """
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
    """
    cmap_lut は Hue と Degree を Index として使用する。
    このうち Hue 一律なのだが、Degree に関しては
    各 Index に対応する Radian 値が Hue によって変動する。
    この Radian の最小値・最大値を計算しておく。
    あとの補間計算で使うので。
    例えば、
      st_degree_l[np.deg2rad(30)] --> -0.4 * pi,
      ed_degree_l[np.deg2rad(30)] --> 0.9 * pi,
      st_degree_l[np.deg2rad(90)] --> -0.7 * pi,
      ed_degree_l[np.deg2rad(90)] --> 1.3 * pi,
    みたいなイメージ。
    """
    lh_inner_lut = np.load(
        get_gamut_boundary_lut_name(inner_color_space_name))
    inner_cusp_l_lut = calc_cusp_lut(lh_lut=lh_inner_lut)
    st_degree_l, ed_degree_l, st_degree_c, ed_degree_c =\
        calc_chroma_map_degree2(l_focal_lut, c_focal_lut, inner_cusp_l_lut)

    """
    cmap_lut から Hue-Degree のペアに該当する
    destination の distance を補間計算で算出する。
    """
    dst_distance_l = interpolate_chroma_map_lut(
        cmap_hd_lut=cmap_lut_l, degree_min=st_degree_l,
        degree_max=ed_degree_l, data_hd=hd_data_l)
    dst_distance_c = interpolate_chroma_map_lut(
        cmap_hd_lut=cmap_lut_c, degree_min=st_degree_c,
        degree_max=ed_degree_c, data_hd=hd_data_c)

    """
    out of gamut ではないデータは処理をしないようにする
    """
    eliminate_inner_gamut_data_l_focal(
        dst_distance=dst_distance_l, src_chroma=src_chroma,
        src_lightness=src_lightness, l_focal=l_focal)
    eliminate_inner_gamut_data_c_focal(
        dst_distance=dst_distance_c, src_chroma=src_chroma,
        src_lightness=src_lightness, c_focal=c_focal)

    """
    このあとの結果統合処理用に Hue-Degree --> Chroma-Lightness に
    変換する。
    """
    dst_chroma_l, dst_lightness_l\
        = calc_chroma_lightness_using_length_from_l_focal(
            distance=dst_distance_l, degree=hd_data_l[..., 1], l_focal=l_focal)
    dst_chroma_c, dst_lightness_c\
        = calc_chroma_lightness_using_length_from_c_focal(
            distance=dst_distance_c, degree=hd_data_c[..., 1], c_focal=c_focal)

    """
    L_Focalベースと C_Focalベースの結果を統合
    """
    dst_chroma, dst_lightness = merge_lightness_mapping(
        hd_data_l=hd_data_l, st_degree_l=st_degree_l,
        chroma_map_l=dst_chroma_l, lightness_map_l=dst_lightness_l,
        chroma_map_c=dst_chroma_c, lightness_map_c=dst_lightness_c)

    return dst_chroma, dst_lightness


def calc_rgb_from_hue_chroma_lightness(
        hue, chroma, lightness,
        color_space_name=cs.BT709):
    aa = chroma * np.cos(hue)
    bb = chroma * np.sin(hue)
    lab = np.dstack((lightness, aa, bb))
    large_xyz = Lab_to_XYZ(lab)
    rgb = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65,
        RGB_COLOURSPACES[color_space_name].XYZ_to_RGB_matrix)

    return rgb


def bt2407_gamut_mapping_for_rgb_linear(
        rgb_linear=(np.array([1023, 512, 256])/1023)**2.4,
        outer_color_space_name=cs.BT2020,
        inner_color_space_name=cs.BT709):
    """
    BT.2407 の Luminance Mapping に基づいた Gamut Mapping をする。
    なお、事前に諸々のLUTを作成しておく必要がある。

    Parameters
    ----------
    rgb_linear : array_like
        RGB のデータ。**Linear** であること。
    outer_color_space_name : str
        src の 色域の Colour Science for Python の名称。
    inner_color_space_name : str
        dst の 色域の Colour Science for Python の名称。

    Returns
    -------
    rgb_dst : array_like
        inner_color_space に **変換済み** の RGB値
        ただし RGB値は **Linear** である。必要に応じて OETF を適用すること。
    """
    # shape を (N, 3) に変更。色々と処理しやすくするため。
    if (len(rgb_linear.shape) != 2) and (rgb_linear.shape[1] != 3):
        src_shape = rgb_linear.shape
        shape_0 = 1
        for shape_val in src_shape[:-1]:
            shape_0 *= shape_val
        src_rgb = rgb_linear.reshape((shape_0, 3))
        restore_shape_flag = True
    else:
        restore_shape_flag = False
        src_rgb = rgb_linear

    # Load LUTs
    _, l_focal_lut, c_focal_lut = load_cusp_focal_lut(
        outer_color_space_name=outer_color_space_name,
        inner_color_space_name=inner_color_space_name)

    # Non-Linear RGB --> Hue Degree への変換
    hd_data_l, hd_data_c, src_chroma, src_lightness\
        = calc_hue_degree_data_from_rgb(
            rgb_linear=src_rgb, l_focal_lut=l_focal_lut,
            c_focal_lut=c_focal_lut,
            outer_color_space_name=outer_color_space_name)

    # Luminance Mapping
    dst_chroma, dst_lightness = luminance_mapping_in_hd_space(
        hd_data_l=hd_data_l, hd_data_c=hd_data_c,
        src_chroma=src_chroma, src_lightness=src_lightness,
        l_focal_lut=l_focal_lut, c_focal_lut=c_focal_lut)

    # Hue Degree --> Non-Linear RGB へ戻す
    rgb_dst = calc_rgb_from_hue_chroma_lightness(
        hue=hd_data_l[..., 0], chroma=dst_chroma, lightness=dst_lightness,
        color_space_name=inner_color_space_name)

    rgb_dst = np.clip(rgb_dst, 0.0, 1.0)

    if restore_shape_flag:
        return rgb_dst.reshape((src_shape))
    else:
        return rgb_dst[0]


def main_func():
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
