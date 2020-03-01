#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# 概要
評価用のテストパターン作成ツール集

# 使い方

"""

import sys
import numpy as np
from scipy import linalg
import common
import plot_utility as pu
import matplotlib.pyplot as plt
# from PIL import Image

const_lab_delta = 6.0/29.0
const_lab_xn_d65 = 95.047
const_lab_yn_d65 = 100.0
const_lab_zn_d65 = 108.883

""" src : http://www.easyrgb.com/en/math.php """
const_d50_large_xyz = [96.422, 100.000, 82.521]

const_d65_large_xyz = [95.047, 100.000, 108.883]
const_d60_large_xyz = [95.26460746, 100.0, 100.88251844]

const_sRGB_xy = [[0.64, 0.33],
                 [0.30, 0.60],
                 [0.15, 0.06]]

const_AdobeRGB_xy = [[0.6400, 0.3300],
                     [0.2100, 0.7100],
                     [0.1500, 0.0600]]

const_ntsc_xy = [[0.67, 0.33],
                 [0.21, 0.71],
                 [0.14, 0.08],
                 [0.310, 0.316]]

const_rec601_xy = const_ntsc_xy

const_rec709_xy = const_sRGB_xy

const_dci_p3_xy = [[0.680, 0.320],
                   [0.265, 0.690],
                   [0.150, 0.060]]

const_rec2020_xy = [[0.708, 0.292],
                    [0.170, 0.797],
                    [0.131, 0.046]]

const_sRGB_xyz = [[0.64, 0.33, 0.03],
                  [0.30, 0.60, 0.10],
                  [0.15, 0.06, 0.79]]

const_s_gamut3_cine_xy = [[0.766, 0.275],
                          [0.225, 0.800],
                          [0.089, -0.087]]

const_s_gamut3_xy = [[0.730, 0.280],
                     [0.140, 0.855],
                     [0.100, -0.050]]

const_aces_ap0_xy = [[0.73470, 0.26530],
                     [0.00000, 1.00000],
                     [0.00010, -0.07700]]

const_aces_ap1_xy = [[0.713, 0.293],
                     [0.165, 0.830],
                     [0.128, 0.044]]

const_xyz_to_lms = [[0.8951000, 0.2664000, -0.1614000],
                    [-0.7502000, 1.7135000, 0.0367000],
                    [0.0389000, -0.0685000, 1.0296000]]

const_rgb_to_large_xyz = [[2.7689, 1.7517, 1.1302],
                          [1.0000, 4.5907, 0.0601],
                          [0.0000, 0.0565, 5.5943]]

const_aces_white_xy = [0.32168, 0.33767]

const_d65_xy = [0.31271, 0.32902]
const_d50_xy = [0.34567, 0.35850]
const_dci_white_xy = [0.314, 0.351]
const_dci_white_xyz = [const_dci_white_xy[0],
                       const_dci_white_xy[1],
                       1 - const_dci_white_xy[0] - const_dci_white_xy[1]]

const_rec601_y_coef = [0.2990, 0.5870, 0.1140]
const_rec709_y_coef = [0.2126, 0.7152, 0.0722]

const_srgb_eotf_threshold = 0.04045
const_srgb_oetf_threshold = 0.0031308

rgb2yuv_rec709mtx = [[0.2126, 0.7152, 0.0722],
                     [-0.2126/1.8556, -0.7152/1.8556, 0.9278/1.8556],
                     [0.7874/1.5748, -0.7152/1.5748, -0.0722/1.5748]]
rgb2yuv_rec2020mtx = [[0.2627, 0.6780, 0.0593],
                      [-0.2627/1.8814, -0.6780/1.8814, (1-0.0593)/1.8814],
                      [(1-0.2627)/1.4746, -0.6780/1.4746, -0.0593/1.4746]]


def get_white_point_conv_matrix(src=const_d65_large_xyz,
                                dst=const_d50_large_xyz):
    """
    # brief
    execute bradford transform.
    # reference
    http://w3.kcua.ac.jp/~fujiwara/infosci/colorspace/bradford.html
    """

    src = np.array(src)
    dst = np.array(dst)

    # LMS値を求めよう
    # --------------------------------------
    ma = np.array(const_xyz_to_lms)
    ma_inv = linalg.inv(ma)

    src_LMS = ma.dot(src)
    dst_LMS = ma.dot(dst)

    # M行列を求めよう
    # --------------------------------------
    mtx = [[dst_LMS[0]/src_LMS[0], 0.0, 0.0],
           [0.0, dst_LMS[1]/src_LMS[1], 0.0],
           [0.0, 0.0, dst_LMS[2]/src_LMS[2]]]

    m_mtx = ma_inv.dot(mtx).dot(ma)

    return m_mtx


def xy_to_xyz_internal(xy):
    rz = 1 - (xy[0][0] + xy[0][1])
    gz = 1 - (xy[1][0] + xy[1][1])
    bz = 1 - (xy[2][0] + xy[2][1])

    xyz = [[xy[0][0], xy[0][1], rz],
           [xy[1][0], xy[1][1], gz],
           [xy[2][0], xy[2][1], bz]]

    return xyz


def get_rgb_to_xyz_matrix(gamut=const_sRGB_xy, white=const_d65_large_xyz):

    # まずは xyz 座標を準備
    # ------------------------------------------------
    if np.array(gamut).shape == (3, 2):
        gamut = xy_to_xyz_internal(gamut)
    elif np.array(gamut).shape == (3, 3):
        pass
    else:
        print("============ Fatal Error ============")
        print("invalid xy gamut parameter.")
        print("=====================================")
        sys.exit(1)

    gamut_mtx = np.array(gamut)

    # 白色点の XYZ を算出。Y=1 となるように調整
    # ------------------------------------------------
    large_xyz = [white[0] / white[1], white[1] / white[1], white[2] / white[1]]
    large_xyz = np.array(large_xyz)

    # Sr, Sg, Sb を算出
    # ------------------------------------------------
    s = linalg.inv(gamut_mtx[0:3]).T.dot(large_xyz)

    # RGB2XYZ 行列を算出
    # ------------------------------------------------
    s_matrix = [[s[0], 0.0,  0.0],
                [0.0,  s[1], 0.0],
                [0.0,  0.0,  s[2]]]
    s_matrix = np.array(s_matrix)
    rgb2xyz_mtx = gamut_mtx.T.dot(s_matrix)

    return rgb2xyz_mtx


def color_cvt(img, mtx):
    """
    # 概要
    img に対して mtx を適用する。
    # 注意事項
    例によって、RGBの並びを考えている。BGRの並びの場合は
    img[:, :, ::-1] してから関数をコールすること。
    """
    try:
        img_max = np.iinfo(img.dtype).max
        img_min = np.iinfo(img.dtype).min
    except ValueError:
        img_max = np.finfo(img.dtype).max
        img_min = np.finfo(img.dtype).min

    r, g, b = np.dsplit(img, 3)
    ro = r * mtx[0][0] + g * mtx[0][1] + b * mtx[0][2]
    go = r * mtx[1][0] + g * mtx[1][1] + b * mtx[1][2]
    bo = r * mtx[2][0] + g * mtx[2][1] + b * mtx[2][2]

    out_img = np.dstack((ro, go, bo))

    out_img[out_img < img_min] = img_min
    out_img[out_img > img_max] = img_max

    return out_img


def xyY_to_XYZ(xyY):
    """
    # 概要
    xyYからXYZを計算する

    # 入力データ
    numpy形式。shape = (1, N, 3)
    """
    small_x, small_y, large_y = np.dsplit(xyY, 3)
    small_z = 1 - small_x - small_y
    small_yx = small_y / small_x
    small_yz = small_y / small_z
    if np.sum(small_yx == 0) or np.sum(small_yz == 0):
        print("WARNING")
        print("    ZERO DIV ERROR IS PREVENTED at xyY_to_XYZ()")

    small_yx[small_yx == 0] = 1
    small_yz[small_yz == 0] = 1

    large_x = large_y / small_yx
    large_z = large_y / small_yz

    return np.dstack((large_x, large_y, large_z))


def srgb_to_linear(img):
    """
    # 概要
    sRGB の 画像をリニアに戻す

    # 注意事項
    img は float型であること
    """

    float_list = [np.float, np.float16, np.float32]

    if img.dtype not in float_list:
        raise TypeError('img must be float type!')

    if np.sum(img > 1) > 0:
        raise ValueError('img must be normalized to 0 .. 1')

    lower_img = img / 12.92
    upper_img = ((img + 0.055) / 1.055) ** 2.4

    out_img = lower_img * (img <= const_srgb_eotf_threshold)\
        + upper_img * (img >= const_srgb_eotf_threshold)

    return out_img


def linear_to_pq(img):
    """
    # brief
    convert linear data to pq.
    # reference
    ITU-R BT.2100-0
    """
    float_list = [np.float, np.float16, np.float32]

    if img.dtype not in float_list:
        raise TypeError('img must be float type!')

    if np.sum(img > 1) > 0:
        raise ValueError('img must be normalized to 0 .. 1')

    m1 = 2610 / 16384
    m2 = 2523 / 4096 * 128
    c1 = 3424 / 4096
    c2 = 2413 / 4096 * 32
    c3 = 2392 / 4096 * 32

    numerator = c1 + c2 * (img**m1)
    denominator = 1 + c3 * (img**m1)

    return (numerator / denominator) ** m2


def pq_to_linear(img):
    """
    # brief
    convert pq data to linear data.
    # reference
    ITU-R BT.2100-0
    """

    m1 = 2610 / 16384
    m2 = 2523 / 4096 * 128
    c1 = 3424 / 4096
    c2 = 2413 / 4096 * 32
    c3 = 2392 / 4096 * 32

    numerator = (img ** (1/m2)) - c1
    numerator[numerator < 0] = 0
    denominator = c2 - c3 * (img ** (1/m2))

    return (numerator / denominator) ** (1/m1)


def linear_to_hlg(img):
    """
    # brief
    convert linear data to hlg.
    # reference
    ITU-R BT.2100-0
    """
    a = 0.17883277
    b = 0.02372241
    c = 1.00429347
    lower = (img <= 1/12) * ((3 * img) ** (0.5))
    img_b = img - b
    img_b[img_b < 0] = 0.00001
    upper = (img > 1/12) * (a * np.log(img_b) + c)
    return lower + upper


def linear_to_srgb(img):
    """
    # 概要
    リニアの画像をsRGBに変換する

    # 注意事項
    img は float型であること
    """

    float_list = [np.float, np.float16, np.float32]

    if img.dtype not in float_list:
        raise TypeError('img must be float type!')

    if np.sum(img > 1) > 0:
        raise ValueError('img must be normalized to 0 .. 1')

    lower_img = img * 12.92
    upper_img = 1.055 * (img ** (1/2.4)) - 0.055

    out_img = lower_img * (img <= const_srgb_oetf_threshold)\
        + upper_img * (img >= const_srgb_oetf_threshold)

    return out_img


def linear_to_rec709(img, plot=False):
    """
    # 概要
    REC709でガンマを掛けます

    # 参考資料
    https://en.wikipedia.org/wiki/Rec._709
    """
    threshold = 0.018
    if np.sum(img > 1) > 0:
        raise ValueError('img must be normalized to 0 .. 1')

    lower_img = 4.5 * img
    upper_img = (1.099 * (img ** 0.45)) - 0.099

    out_img = lower_img * (img < threshold)\
        + upper_img * (img >= threshold)

    if plot:
        ax = pu.plot_1_graph()
        ax.plot(out_img.flatten())
        plt.show()

    return out_img


def rec709_to_linear(img, plot=False):
    """
    # 概要
    REC709のガンマを解除してLinearな特性にします

    # 参考資料
    https://en.wikipedia.org/wiki/Rec._709
    """
    threshold = 0.081
    if np.sum(img > 1) > 0:
        raise ValueError('img must be normalized to 0 .. 1')

    lower_img = img / 4.5
    upper_img = ((img + 0.099) / 1.099) ** (1/0.45)

    out_img = lower_img * (img < threshold)\
        + upper_img * (img >= threshold)

    if plot:
        ax = pu.plot_1_graph()
        ax.plot(out_img.flatten())
        plt.show()

    return out_img


def xyY_to_RGB(xyY, gamut=const_sRGB_xy, white=const_d65_large_xyz):
    """
    # 概要
    xyY から RGB値を算出する
    # 入力データ
    numpy形式。shape = (N, M, 3)
    """
    if not common.is_img_shape(xyY):
        raise TypeError('xyY shape must be (N, M, 3)')

    max_large_y = np.max(xyY[:, :, 2])
    large_xyz = xyY_to_XYZ(xyY)

    # 正規化。俺のモジュールは [0:1] で入力する前提のため
    large_xyz = large_xyz / max_large_y
    rgb = large_xyz_to_rgb(large_xyz=large_xyz, gamut=gamut, white=white)
    return rgb


def rgb_to_large_xyz(rgb, gamut=const_sRGB_xy,
                     white=const_d65_large_xyz):
    """
    # 概要
    RGB値 から XYZ値を算出する
    # 入力データ
    numpy形式。shape = (N, M, 3)
    ガンマがキャンセルされ、かつ 0:1 に正規化されていること。
    """
    if not common.is_img_shape(rgb):
        raise TypeError('XYZ shape must be (N, M, 3)')
    cvt_mtx = get_rgb_to_xyz_matrix(gamut=gamut, white=white)
    large_xyz = color_cvt(rgb, cvt_mtx)

    if np.sum(large_xyz < 0) > 0:
        print("function : rgb_to_large_xyz")
        print("  underflow has occured!")

    large_xyz[large_xyz < 0] = 0

    return large_xyz


def large_xyz_to_rgb(large_xyz, gamut=const_sRGB_xy,
                     white=const_d65_large_xyz):
    """
    # 概要
    XYZ から RGB値を算出する
    # 入力データ
    numpy形式。shape = (N, M, 3)
    """
    if not common.is_img_shape(large_xyz):
        raise TypeError('XYZ shape must be (N, M, 3)')
    cvt_mtx = get_rgb_to_xyz_matrix(gamut=gamut, white=white)
    cvt_mtx = linalg.inv(cvt_mtx)
    # rgb = color_cvt(large_xyz, cvt_mtx) / white[1]

    rgb = color_cvt(large_xyz, cvt_mtx)

    if (np.sum(rgb < 0) > 0) or (np.sum(rgb > 1) > 0):
        print("function : large_xyz_to_rgb")
        print("  overflow or underflow has occured!")

    rgb[rgb < 0] = 0
    rgb[rgb > 1] = 1

    return rgb


def _func_t(t):
    threshold = const_lab_delta ** 3
    upper = (t > threshold) * (t ** (1/3))
    lower = (t <= threshold) * (t/(3 * (const_lab_delta ** 2)) + 4/29)
    return upper + lower


def _func_t_inverse(t):
    upper = (t > const_lab_delta) * (t ** 3)
    lower = (t <= const_lab_delta) * 3 * (const_lab_delta ** 2) * (t - 4/29)
    return upper + lower


def lab_star_to_large_xyz(lab, white=const_d50_large_xyz):
    """
    # 概要
    L*a*b* から XYZ値を算出する
    # 入力データ
    numpy形式。shape = (N, M, 3)
    # 参考
    https://en.wikipedia.org/wiki/Lab_color_space
    """
    if not common.is_img_shape(lab):
        raise TypeError('lab shape must be (N, M, 3)')

    l, a, b = np.dsplit(lab, 3)
    white = [x / white[1] for x in white]
    large_x = white[0] * _func_t_inverse((l + 16)/116 + a/500)
    large_y = white[1] * _func_t_inverse((l + 16)/116)
    large_z = white[2] * _func_t_inverse((l + 16)/116 - b/200)

    return np.dstack((large_x, large_y, large_z))


def large_xyz_to_lab_star(large_xyz, white=const_d50_large_xyz):
    """
    # 概要
    L*a*b* から XYZ値を算出する
    # 入力データ
    numpy形式。shape = (N, M, 3)
    # 参考
    https://en.wikipedia.org/wiki/Lab_color_space
    """
    if not common.is_img_shape(large_xyz):
        raise TypeError('large_xyz shape must be (N, M, 3)')

    x, y, z = np.dsplit(large_xyz, 3)
    white = [x / white[1] for x in white]
    l = 116 * _func_t(y/white[1]) - 16
    a = 500 * (_func_t(x/white[0]) - _func_t(y/white[1]))
    b = 200 * (_func_t(y/white[1]) - _func_t(z/white[2]))

    return np.dstack((l, a, b))


def is_inside_gamut(xy, gamut=const_rec2020_xy):
    """
    # 概要
    xy座標が gamut 内部にあるか判別する
    # In/Out
    xy は (N, 2) の numpy 配列であること
    # 参考
    http://www.sousakuba.com/Programming/gs_hittest_point_triangle.html
    """
    # parameter check
    # -----------------------------------------
    gamut = np.array(gamut.copy())

    if not common.is_small_xy_array_shape(xy):
        print('parameer "xy" is invalid.')
        return False

    if not common.is_small_xy_array_shape(gamut):
        print('parameer "gamut" is invalid.')
        return False

    if gamut.shape[0] != 3:
        print('parameer "gamut" is invalid.')
        return False

    # calc vector
    # -----------------------------------------
    rg = gamut[1] - gamut[0]
    gw = xy - gamut[1]
    gb = gamut[2] - gamut[1]
    bw = xy - gamut[2]
    br = gamut[0] - gamut[2]
    rw = xy - gamut[0]

    r_result = np.cross(rg, gw)
    g_result = np.cross(gb, bw)
    b_result = np.cross(br, rw)

    result = (r_result >= 0) & (g_result >= 0) & (b_result >= 0)

    return result


def small_xy_to_xyY(xy, large_y):
    if not common.is_small_xy_array_shape(xy):
        print('parameer "xy" is invalid.')
        return False

    x = xy[:, 0]
    y = xy[:, 1]
    large_y = np.ones_like(x) * large_y
    xyY = np.dstack((x, y, large_y))

    return xyY


def large_xyz_to_small_xy(large_xyz):
    large_x, large_y, large_z = np.dsplit(large_xyz, 3)
    xyz_sum = large_x + large_y + large_z

    if np.sum(xyz_sum == 0) > 0:
        print("Warning:")
        print("  ZERO DIVISION HAS OCCURED at large_xyz_to_small_xy()")
        xyz_sum[xyz_sum == 0] = 1

    small_x = large_x / xyz_sum
    small_y = large_y / xyz_sum

    return np.dstack((small_x, small_y))


def extract_profile(in_name, out_name):
    # filename = in_name
    # img = Image.open(filename)
    # out_file_name = out_name
    # # print(img.info.keys())

    # with open(out_file_name, 'wb') as f:
    #     f.write(img.info['icc_profile'])
    pass


def large_xyz_to_uv_dash(large_xyz):
    """
    # 概要
    XYZ から u'v' を計算する。
    # 参考
    https://en.wikipedia.org/wiki/CIELUV
    """
    large_x, large_y, large_z = np.dsplit(large_xyz, 3)
    denominator = large_x + (15 * large_y) + (3 * large_z)
    if np.sum(denominator == 0) > 0:
        print("Warning:")
        print("  ZERO DIVISION HAS OCCURED at large_xyz_to_uv_dash()")
        denominator[denominator == 0] = 1
    u_dash = (4 * large_x) / denominator
    v_dash = (9 * large_y) / denominator

    return np.dstack((u_dash, v_dash))


def small_xy_to_uv_dash(small_xy):
    """
    # 概要
    xy から u'v' を計算する。
    # 参考
    https://en.wikipedia.org/wiki/CIELUV
    """
    x, y = np.dsplit(small_xy, 2)
    denominator = (-2 * x) + (12 * y) + 3
    if np.sum(denominator == 0) > 0:
        print("Warning:")
        print("  ZERO DIVISION HAS OCCURED at small_xy_to_uv_dash()")
        denominator[denominator == 0] = 1
    u_dash = (4 * x) / denominator
    v_dash = (9 * y) / denominator

    return np.dstack((u_dash, v_dash))


def uv_dash_to_small_xy(uv):
    """
    # 概要
    u'v' から xy を計算する。
    # 参考
    https://en.wikipedia.org/wiki/CIELUV
    """
    u_dash, v_dash = np.dsplit(uv)
    denominator = (6 * u_dash) + (-16 * v_dash) + 12
    if np.sum(denominator == 0) > 0:
        print("Warning:")
        print("  ZERO DIVISION HAS OCCURED at small_xy_to_uv_dash()")
        denominator[denominator == 0] = 1
    x = (9 * u_dash) / denominator
    y = (4 * v_dash) / denominator

    return np.dstack((x, y))


def large_xyz_to_luv_star(large_xyz, white_xyz):
    """
    # 概要
    XYZ から L*u*v* を計算する。
    # 参考
    https://en.wikipedia.org/wiki/CIELUV
    """
    large_x, large_y, large_z = np.dsplit(large_xyz, 3)
    white_xyz = np.array(white_xyz)
    white_xyz = (white_xyz / white_xyz[1]).reshape((1, 1, 3))
    x_n, y_n, z_n = np.dsplit(white_xyz, 3)
    threshold = (6/29) ** 3
    judge = (large_y / y_n)
    l_lower = (judge <= threshold) * (((29/3) ** 3) * (large_y / y_n))
    l_upper = (judge > threshold) * (116 * ((large_y / y_n) ** (1/3)) - 16)
    l_star = l_lower + l_upper

    u_dash, v_dash = np.dsplit(large_xyz_to_uv_dash(large_xyz), 2)
    u_n_dash, v_n_dash = np.dsplit(large_xyz_to_uv_dash(white_xyz), 2)
    u_star = 13 * l_star * (u_dash - u_n_dash)
    v_star = 13 * l_star * (v_dash - v_n_dash)

    return np.dstack((l_star, u_star, v_star))


def luv_star_to_large_xyz(luv_star, white_xyz):
    """
    # 概要
    L*u*v* から XYZ を計算する。
    # 参考
    https://en.wikipedia.org/wiki/CIELUV
    """
    l_star, u_star, v_star = np.dsplit(luv_star, 3)
    white_xyz = np.array(white_xyz)
    white_xyz = (white_xyz / white_xyz[1]).reshape((1, 1, 3))
    u_n_dash, v_n_dash = np.dsplit(large_xyz_to_uv_dash(white_xyz), 2)
    if np.sum(l_star == 0) > 0:
        print("Warning:")
        print("  ZERO DIVISION HAS OCCURED at luv_star_to_large_xyz()")
        l_star[l_star == 0] = 1
    u_dash = (u_star / (13 * l_star)) + u_n_dash
    v_dash = (v_star / (13 * l_star)) + v_n_dash

    """
    u', v' は white が適切なら 0 にならないので ZERO DIVISION の
    チェックしません(手抜き)
    """
    x_n, y_n, z_n = np.dsplit(white_xyz, 3)
    y_lower = (l_star <= 8) * (y_n * l_star * ((3/29)**3))
    y_upper = (l_star > 8) * (y_n * (((l_star + 16)/116)**3))
    large_y = y_lower + y_upper
    large_x = large_y * (9 * u_dash) / (4 * v_dash)
    large_z = large_y * (12 + (-3 * u_dash) + (-20 * v_dash)) / (4 * v_dash)

    return np.dstack((large_x, large_y, large_z))


def ycbcr_to_yuv(ycbcr, bit_depth=10):
    """
    Converts from YCbCr(Limited Range) to YUV(float).

    Parameters
    ----------
    ycbcr : numeric or ndarray. shape must be (n, 3)
        ycbcr data.

    bit_depth : integer
        bit depth of the ycbcr.

    Returns
    -------
    numeric or ndarray
        YUV value

    Notes
    -----
    -

    Examples
    --------
    >>> ycbcr = np.array([[381, 470, 578]])
    >>> yuv = ycbcr_to_yuv(ycbcr, bit_depth=10)
    """

    bit_multi = 2 ** (bit_depth - 8)
    y_coef = 219 * bit_multi
    y_offset = 16 * bit_multi
    cbcr_coef = 224 * bit_multi
    cbcr_offset = 128 * bit_multi

    ycbcr_tmp = ycbcr.copy()
    ycbcr_tmp[:, 0] = (ycbcr_tmp[:, 0] - y_offset) / y_coef
    ycbcr_tmp[:, 1] = (ycbcr_tmp[:, 1] - cbcr_offset) / cbcr_coef
    ycbcr_tmp[:, 2] = (ycbcr_tmp[:, 2] - cbcr_offset) / cbcr_coef

    return ycbcr_tmp


def yuv_to_ycbcr(yuv, bit_depth=10):
    """
    Converts from YUV(float) to YCbCr(Limited Range).

    Parameters
    ----------
    yuv : numeric or ndarray. shape must be (n, 3)
        yuv data.

    bit_depth : integer
        bit depth of the ycbcr.

    Returns
    -------
    numeric or ndarray
        YCbCr value

    Notes
    -----
    -

    Examples
    --------
    >>> yuv = np.array([[1.0, 0.0, 0.0]])
    >>> yuv = ycbcr_to_yuv(ycbcr, bit_depth=10)
    """

    bit_multi = 2 ** (bit_depth - 8)
    y_coef = 219 * bit_multi
    y_offset = 16 * bit_multi
    cbcr_coef = 224 * bit_multi
    cbcr_offset = 128 * bit_multi

    ycbcr = yuv.copy()
    ycbcr[:, 0] = np.round(ycbcr[:, 0] * y_coef + y_offset)
    ycbcr[:, 1] = np.round(ycbcr[:, 1] * cbcr_coef + cbcr_offset)
    ycbcr[:, 2] = np.round(ycbcr[:, 2] * cbcr_coef + cbcr_offset)

    return ycbcr


if __name__ == '__main__':
    # lab = np.ones((1, 1, 3))
    # lab[0][0][0] = 42.101
    # lab[0][0][1] = 53.378
    # lab[0][0][2] = 28.19
    # # lab[0][0][0] = 96.539
    # # lab[0][0][1] = -0.425
    # # lab[0][0][2] = 1.186
    # large_xyz = lab_to_large_xyz(lab, white=const_d50_large_xyz)
    # print(large_xyz)
    # d50_to_d65_mtx = get_white_point_conv_matrix(src=const_d50_large_xyz,
    #                                              dst=const_d65_large_xyz)
    # large_xyz = color_cvt(img=large_xyz, mtx=d50_to_d65_mtx)
    # print(large_xyz)
    # rgb = large_xyz_to_rgb(large_xyz, const_sRGB_xy)
    # print(rgb)
    # print(linear_to_srgb(rgb/100) * 255)
    # print(get_rgb_to_xyz_matrix(gamut=const_sRGB_xy, white=const_d65_large_xyz))
    # rgb = np.ones((1, 1, 3))
    # rgb[0][0][0] = 243
    # rgb[0][0][1] = 243
    # rgb[0][0][2] = 242
    # rgb = srgb_to_linear(rgb/255)
    # xyz = rgb_to_large_xyz(rgb=rgb, gamut=const_sRGB_xy,
    #                        white=const_d65_large_xyz)
    # xyz *= 100
    # mtx = get_white_point_conv_matrix(src=const_d65_large_xyz,
    #                                   dst=const_d50_large_xyz)
    # xyz = color_cvt(img=xyz, mtx=mtx)
    # print(xyz)
    # lab = large_xyz_to_lab(large_xyz=xyz, white=const_d50_large_xyz)
    # print(lab)
    # out = linear_to_rec709(np.linspace(0, 1, 1024), plot=True)
    # rec709_to_linear(out, plot=True)
    # extract_profile("C:\home\DSC00419.jpg", "C:\home\DSC00419.icc")
    pass
