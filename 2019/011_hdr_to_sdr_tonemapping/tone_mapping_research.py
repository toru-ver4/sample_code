#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sympy import symbols, solve
from sympy.utilities.lambdify import lambdify
from colour import xyY_to_XYZ, XYZ_to_RGB, RGB_to_XYZ, XYZ_to_xyY, RGB_to_RGB
from colour.models import BT2020_COLOURSPACE, BT709_COLOURSPACE
import OpenImageIO as oiio
from colour.colorimetry import CMFS, ILLUMINANTS
from colour.models import eotf_BT1886


# 自作ライブラリのインポート
import TyImageIO as tyio
import transfer_functions as tf
import plot_utility as pu
import test_pattern_generator2 as tpg
import color_space as cs

src_exr = "./img/SMPTE ST2084_ITU-R BT.2020_D65_1920x1080_rev03_type1.exr"
dst_exr = "./img/SMPTE ST2084_ITU-R BT.2020_D65_1920x1080_rev03_type1_c.exr"
after_3dlut_exr = "./img/after_3dlut_with_pq_oetf.exr"
sdr_eotf = tf.SRGB

CMFS_NAME = 'CIE 1931 2 Degree Standard Observer'
D65_WHITE = ILLUMINANTS[CMFS_NAME]['D65']
UNIVERSAL_COLOR_LIST = ["#F6AA00", "#FFF100", "#03AF7A",
                        "#005AFF", "#4DC4FF", "#804000"]
st_pos_h = 57
st_pos_v = 289
h_sample = 1024


def dump_attr(fname, attr):
    print("attr data of {} is as follows.".format(fname))
    print(attr)


def dump_img_info(img):
    print("shape: ", img.shape)
    print("dtype: ", img.dtype)
    print("min: ", np.min(img))
    print("max: ", np.max(img))


def correct_pq_exr_gain():
    """
    exrファイルだと PQ のピークが 100.0 なので
    1.0 になるように正規化する。
    """
    reader = tyio.TyReader(src_exr)
    img = reader.read()
    dump_attr(src_exr, reader.get_attr())
    dump_img_info(img)
    writer = tyio.TyWriter(img / 100.0, dst_exr)
    writer.write(out_img_type_desc=oiio.FLOAT)


def check_after_3dlut_exr():
    """
    3DLUT通過後のデータを確認
    """
    reader = tyio.TyReader(after_3dlut_exr)
    img = reader.read()
    dump_attr(after_3dlut_exr, reader.get_attr())
    dump_img_info(img)


def plot_eetf(eetf):
    """
    EETF のプロット
    """
    x = np.linspace(0, 1, h_sample)
    y_simulation = youtube_tonemapping(x, 400, 1000)
    y = eetf[st_pos_v, st_pos_h:st_pos_h+h_sample, 1]
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Tone Mapping Characteristics",
        graph_title_size=None,
        xlabel="ST2084 Code Value",
        ylabel="ST2084 Code Value",
        axis_label_size=None,
        legend_size=17,
        xtick_size=19,
        ytick_size=19,
        xlim=None,
        ylim=None,
        xtick=[0.1 * x for x in range(11)],
        ytick=[0.05 * x for x in range(11)],
        linewidth=8,
        minor_xtick_num=None,
        minor_ytick_num=None)
    # ax1.set_xscale('log', basex=10.0)
    # ax1.set_yscale('log', basey=10.0)
    # ax1.plot(x_luminance, y_luminance, 'o', label="YouTube HDR to SDR EETF")
    ax1.plot(x, y, '-', label="Estimated EETF")
    # ax1.plot(x, y_18, '-', label="18% gray to 100% white")
    ax1.plot(x, y_simulation, '-', label="Estimated Formula",
             lw=3.5)
    # x_val = [1.0 * (10 ** (x - 4)) for x in range(9)]
    # x_caption = [r"$10^{{{}}}$".format(x - 4) for x in range(9)]
    # plt.xticks(x_val, x_caption)
    # y_caption = [r"$10^{{{}}}$".format(x - 4) for x in range(9)]
    # y_val = [1.0 * (10 ** (x - 4)) for x in range(9)]
    # plt.yticks(y_val, y_caption)
    plt.legend(loc='upper left')
    plt.savefig(
        "./blog_img/eetf_codevalue_with_approximation.png",
        bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_eetf_luminance(eetf):
    """
    EETF のプロットを輝度値ベースで
    """
    img = cv2.imread("./img/youtubed_sdr_tp.png")
    y_ref = ((img[3, :, 1] / 255.0) ** 2.4) * 100
    x = np.linspace(0, 1, h_sample)
    y_simulation = youtube_tonemapping(x, 400, 1000)
    y = eetf[st_pos_v, st_pos_h:st_pos_h+h_sample, 1]
    y_sim_luminance = tf.eotf_to_luminance(y_simulation, tf.ST2084)
    y_luminance = tf.eotf_to_luminance(y, tf.ST2084)
    x_luminance = tf.eotf_to_luminance(x, tf.ST2084)
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Tone Mapping Characteristics",
        graph_title_size=None,
        xlabel="Luminance [cd/m2]",
        ylabel="Luminance [cd/m2]",
        axis_label_size=None,
        legend_size=17,
        xtick_size=19,
        ytick_size=19,
        xlim=(-75, 1200),
        ylim=(-5, 105),
        xtick=[x * 100 for x in range(13)],
        ytick=[x * 10 for x in range(11)],
        linewidth=8,
        minor_xtick_num=None,
        minor_ytick_num=None)
    # ax1.set_xscale('log', basex=10.0)
    # ax1.set_yscale('log', basey=10.0)
    # ax1.plot(x_luminance, y_luminance, 'o', label="YouTube HDR to SDR EETF")
    ax1.plot(x_luminance, y_luminance, '-', label="Estimated EETF")
    ax1.plot(x_luminance, y_sim_luminance, '-', label="Estimated Formula",
             lw=3.5)
    # ax1.plot(x_luminance, y_ref, 'o', label="youtube")
    # x_val = [1.0 * (10 ** (x - 4)) for x in range(9)]
    # x_caption = [r"$10^{{{}}}$".format(x - 4) for x in range(9)]
    # plt.xticks(x_val, x_caption)
    # y_caption = [r"$10^{{{}}}$".format(x - 4) for x in range(9)]
    # y_val = [1.0 * (10 ** (x - 4)) for x in range(9)]
    # plt.yticks(y_val, y_caption)
    plt.legend(loc='upper left')
    plt.savefig(
        "./blog_img/eetf_luminance_with_approximation.png",
        bbox_inches='tight', pad_inches=0.1)
    plt.show()


def dump_eetf_info(eetf):
    """
    18% Gray, 100% White などの情報をダンプする
    """

    x_18_gray = tf.oetf_from_luminance(18.0, tf.ST2084)
    x_100_white = tf.oetf_from_luminance(100.0, tf.ST2084)
    x_ref_white = tf.oetf_from_luminance(250.0, tf.ST2084)
    x_18_idx = int(np.round(x_18_gray * 1023))
    x_100_idx = int(np.round(x_100_white * 1023))
    x_ref_idx = int(np.round(x_ref_white * 1023))
    print(x_18_gray, x_100_white, x_ref_white)
    print(x_18_idx, x_100_idx, x_ref_idx)
    y = eetf[st_pos_v, st_pos_h:st_pos_h+h_sample, 1]
    print("18 Gray = {}".format(y[x_18_idx]))
    print("100 white = {}".format(y[x_100_idx]))
    print("250 white = {}".format(y[x_ref_idx]))
    a = (y[x_100_idx] - y[x_18_idx]) / (x_100_white - x_18_gray)
    b = a * -x_18_gray + y[x_18_idx]
    print(a)
    print(b)
    a = 0.74
    b = a * -x_100_white + y[x_100_idx]
    print(a)
    print(b)
    print(tf.eotf_to_luminance(0.0117522745451, tf.ST2084))
    print(tf.eotf_to_luminance(0.3877303064680016, tf.ST2084))


def conv_hdr10_to_sdr_using_formula(
        in_name="./img/test_src_for_youtube_upload_riku.tif"):
    out_name_body = os.path.basename(os.path.splitext(in_name)[0])
    out_name = "./blog_img/{}.png".format(out_name_body)
    img = cv2.imread(
        in_name, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[:, :, ::-1] / 0xFFFF

    img = youtube_tonemapping(img)
    img_linear = tf.eotf_to_luminance(img, tf.ST2084) / 100
    img_linear = RGB_to_RGB(img_linear, BT2020_COLOURSPACE, BT709_COLOURSPACE)
    img_linear = np.clip(img_linear, 0.0, 1.0)
    img = tf.oetf(img_linear, tf.GAMMA24)
    img = np.uint8(np.round(img * 0xFF))
    cv2.imwrite(out_name, img[:, :, ::-1])


def analyze_eetf():
    reader = tyio.TyReader(after_3dlut_exr)
    img = reader.read()
    # img_after_sdr_eotf = tf.eotf(img, sdr_eotf)
    img_after_sdr_eotf = img ** 2.4
    # img_after_sdr_eotf = eotf_BT1886(img, L_B=1/2000)
    img_after_pq_oetf = tf.oetf(img_after_sdr_eotf / 100, tf.ST2084)
    eetf = img_after_pq_oetf
    # dump_eetf_info(eetf)
    # plot_eetf(eetf)
    # plot_eetf_luminance(eetf)
    # plot_apply_eetf_luminance()
    # plot_apply_eetf_code_value()
    conv_hdr10_to_sdr_using_formula(
        in_name="./img/test_src_for_youtube_upload_riku.tif")
    conv_hdr10_to_sdr_using_formula(
        in_name="./img/test_src_for_youtube_upload_umi.tif")
    conv_hdr10_to_sdr_using_formula(
        in_name="./img/test_tp_image_hdr10_tp.tif")
    conv_hdr10_to_sdr_using_formula(
        in_name="./img/test_gamut_tp_1000nits.tif")


def spline_example():
    data = [[0.5, 0.5], [0.8, 1.0], [1.0, 1.0]]
    func = get_top_side_bezier(data)
    x = np.linspace(data[0][0], data[2][0], 256)
    y = func(x)
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 10),
        graph_title="interpolation",
        graph_title_size=None,
        xlabel=None,
        ylabel=None,
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(x, y, 'o', label="Original Sample")
    # ax1.plot(x2, y2_linear, '-', label="linear interpolation")
    # ax1.plot(x2, y2_cubic, '-', label="cubic interpolation")
    plt.legend(loc='upper left')
    # plt.savefig("./blog_img/eetf.png", bbox_inches='tight', pad_inches=0.1)
    plt.show()


def get_top_side_bezier(param=[[0.6, 0.7], [0.8, 1.0], [1.0, 1.0]]):
    """
    Note
    ----
    An example of ```top_param``` is bellow.
        {'x0': 0.5, 'y0': 0.5,
         'x1': 0.7, 'y1': 0.7,
         'x2': 1.0, 'y2': 0.7}
    """
    a_val = param[0][0]
    b_val = param[1][0]
    c_val = param[2][0]
    p_val = param[0][1]
    q_val = param[1][1]
    r_val = param[2][1]

    a, b, c, t, x = symbols('a, b, c, t, x')
    f = (1 - t)**2 * a + 2 * (1 - t) * t * b + t**2 * c - x

    # x について解く
    # ----------------------
    t = solve(f, t)[1]
    t = t.subs({a: a_val, b: b_val, c: c_val})

    # y と t(ここでは u と置いた) の関係式を記述
    # -------------------------------------------
    p, q, r, u, y = symbols('p, q, r, u, y')
    y = (1 - u)**2 * p + 2 * (1 - u) * u * q + u**2 * r

    # パラメータ u と事前に求めた t で置き換える
    # -------------------------------------------
    y = y.subs({p: p_val, q: q_val, r: r_val, u: t})

    func = lambdify(x, y, 'numpy')

    return func


def youtube_linear(x):
    return 0.74 * x + 0.01175


def youtube_tonemapping(x, ks_luminance=400, ke_luminance=1000):
    """
    YouTube の HDR to SDR のトーンマップを模倣してみる。
    中間階調までは直線で、高階調部だけ2次ベジェ曲線で丸める。

    直線の数式は $y = 0.74x + 0.01175$。
    y軸の最大値は 0.508078421517 (100nits)。
    この時の x は $x = (0.508078421517 - 0.01175) / 0.74$ より
    0.6707140831310812 。ちなみに、リニアなら 473.5 nits。

    ks は knee start の略。ke は knee end.
    """
    ks_x = tf.oetf_from_luminance(ks_luminance, tf.ST2084)
    ks_y = youtube_linear(ks_x)
    ke_x = tf.oetf_from_luminance(ke_luminance, tf.ST2084)
    ke_y = tf.oetf_from_luminance(100, tf.ST2084)
    mid_x = 0.6707140831310812
    mid_y = ke_y
    bezie = get_top_side_bezier([[ks_x, ks_y], [mid_x, mid_y], [ke_x, ke_y]])
    y = np.select(
        (x < ks_x, x <= ke_x, x > ke_x),
        (youtube_linear(x), bezie(x), ke_y))

    return y


def test_plot_youtube_tonemapping(ks_luminance=400, ke_luminance=1000):
    x = np.linspace(0, 1, 1024)
    y = youtube_tonemapping(x, ks_luminance, ke_luminance)
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 10),
        graph_title="Tone mapping characteristics",
        graph_title_size=None,
        xlabel="ST2084 Code Value",
        ylabel="ST2084 Code Value",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(x, y, 'o', label="emulation")
    plt.legend(loc='upper left')
    plt.show()


def get_xyY_from_csv(fname="./data/bt2020_xyY_data.csv"):
    data_all = np.loadtxt(fname, delimiter=",", skiprows=1).T
    hue_num = len(data_all[1:]) // 3  # 3 is r, g, b.
    data = data_all[1:]
    xyY_all = []
    for h_idx in range(hue_num):
        xyY = np.dstack((data[h_idx*3], data[h_idx*3+1], data[h_idx*3+2]))
        xyY_all.append(xyY)

    xyY_all = np.array(xyY_all)

    return xyY_all


def make_src_test_pattern(xyY, pixel_num):
    large_xyz = xyY_to_XYZ(xyY)
    xyz_to_rgb_mtx = BT2020_COLOURSPACE.XYZ_to_RGB_matrix
    rgb_linear = XYZ_to_RGB(large_xyz, D65_WHITE, D65_WHITE, xyz_to_rgb_mtx)
    rgb_linear[rgb_linear < 0] = 0.0
    rgb_pq = tf.oetf_from_luminance(rgb_linear * 1000, tf.ST2084)
    rgb_seq = rgb_pq.reshape((1, pixel_num, 3))
    img = np.zeros((1080, 1920, 3), dtype=np.uint16)
    rgb_seq_16bit = np.uint16(np.round(rgb_seq * 0xFFFF))
    img[0, :pixel_num, :] = rgb_seq_16bit
    cv2.imwrite("./img/gamut_check_src.tiff", img[:, :, ::-1])


def restore_dst_test_pattern(fname, pixel_num):
    img = cv2.imread(fname, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[:, :, ::-1]
    rgb_pq = img[0, :pixel_num, :] / 0xFFFF
    rgb_linear = tf.eotf(rgb_pq, tf.GAMMA24)
    rgb_to_xyz_mtx = BT709_COLOURSPACE.RGB_to_XYZ_matrix
    large_xyz = RGB_to_XYZ(rgb_linear, D65_WHITE, D65_WHITE, rgb_to_xyz_mtx)
    xyY = XYZ_to_xyY(large_xyz)
    return xyY


def analyze_gamut_mapping():
    hdr_xyY = get_xyY_from_csv(fname="./data/bt2020_xyY_data.csv")
    pixel_num = hdr_xyY.shape[0] * hdr_xyY.shape[1] * hdr_xyY.shape[2]
    make_src_test_pattern(hdr_xyY, pixel_num)
    sdr_xyY = restore_dst_test_pattern(
        fname="./img/gamut_check_dst.tif", pixel_num=pixel_num)
    plot_xy_move(hdr_xyY.reshape((pixel_num, 3)), sdr_xyY)
    plot_simple_xy_move(hdr_xyY.reshape((pixel_num, 3)), None)


def plot_xy_move(src_xyY, dst_xyY, xmin=0.0, xmax=0.8, ymin=0.0, ymax=0.9):
    """
    src は HDR時の xyY, dst は SDR変換後 の xyY.
    """
    xy_image = tpg.get_chromaticity_image(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    cmf_xy = tpg._get_cmfs_xy()
    xlim = (min(0, xmin), max(0.8, xmax))
    ylim = (min(0, ymin), max(0.9, ymax))
    figsize_h = 8 * 1.0
    figsize_v = 9 * 1.0
    rate = 1.3
    # gamut の用意
    outer_gamut = np.array(tpg.get_primaries(cs.BT2020)[0])
    inner_gamut = np.array(tpg.get_primaries(cs.BT709)[0])
    outer_name = 'ITR-R BT.2020'
    inner_name = 'ITR-R BT.709'
    ax1 = pu.plot_1_graph(fontsize=20 * rate,
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
                          minor_xtick_num=2, minor_ytick_num=2)
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=3.5*rate, label=None)
    ax1.plot((cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
             '-k', lw=3.5*rate, label=None)
    ax1.plot(inner_gamut[:, 0], inner_gamut[:, 1],
             c=UNIVERSAL_COLOR_LIST[0], label=inner_name, lw=2.75*rate)
    ax1.plot(outer_gamut[:, 0], outer_gamut[:, 1],
             c=UNIVERSAL_COLOR_LIST[3], label=outer_name, lw=2.75*rate)
    ax1.plot(tpg.D65_WHITE[0], tpg.D65_WHITE[1], 'x', c='k',
             lw=2.75*rate, label='D65', ms=10*rate, mew=2.75*rate)
    ax1.plot(src_xyY[..., 0], src_xyY[..., 1], ls='', marker='o',
             c='#000000', ms=5.5*rate, label="Before")
    ax1.plot(dst_xyY[..., 0], dst_xyY[..., 1], ls='', marker='+',
             c='#808080', ms=10*rate, mew=2.0*rate,
             label="After")
    # annotation
    arrowprops = dict(
        facecolor='#333333', shrink=0.0, headwidth=6, headlength=8,
        width=1)
    src_xyY = src_xyY.reshape((18, 8, 3))
    dst_xyY = dst_xyY.reshape((18, 8, 3))
    for h_idx in range(18):
        for s_idx in range(8):
            if s_idx < 4:
                continue
            ed_pos = (dst_xyY[h_idx][s_idx][0],
                      dst_xyY[h_idx][s_idx][1])
            st_pos = (src_xyY[h_idx][s_idx][0],
                      src_xyY[h_idx][s_idx][1])
            ax1.annotate("", xy=ed_pos, xytext=st_pos, xycoords='data',
                         textcoords='data', ha='left', va='bottom',
                         arrowprops=arrowprops)

    # for s_idx in range(src_xyY.shape[0]):
    #     ed_pos = (dst_xyY[s_idx][0],
    #               dst_xyY[s_idx][1])
    #     st_pos = (src_xyY[s_idx][0],
    #               src_xyY[s_idx][1])
    #     ax1.annotate("", xy=ed_pos, xytext=st_pos, xycoords='data',
    #                  textcoords='data', ha='left', va='bottom',
    #                  arrowprops=arrowprops)

    ax1.imshow(xy_image, extent=(xmin, xmax, ymin, ymax))
    plt.legend(loc='upper right', fontsize=14 * rate)
    png_file_name = "./blog_img/xyY_plot_hdr_to_sdr.png"
    plt.savefig(png_file_name, bbox_inches='tight')
    plt.show()


def plot_simple_xy_move(
        src_xyY, dst_xyY, xmin=0.0, xmax=0.8, ymin=0.0, ymax=0.9):
    """
    3x3 の Matrix で単純な色域変換をした場合の xy平面でのズレのプロット。
    """
    rgb_2020 = XYZ_to_RGB(xyY_to_XYZ(src_xyY), D65_WHITE, D65_WHITE,
                          BT2020_COLOURSPACE.XYZ_to_RGB_matrix)
    rgb_2020[rgb_2020 < 0] = 0
    rgb_pq = tf.oetf(rgb_2020 * 0.1, tf.ST2084)
    rgb_pq = youtube_tonemapping(rgb_pq)
    rgb_2020 = tf.eotf(rgb_pq, tf.ST2084)
    rgb_709 = RGB_to_RGB(rgb_2020, BT2020_COLOURSPACE, BT709_COLOURSPACE)
    rgb_709 = np.clip(rgb_709, 0.0, 1.0)
    dst_xyY = XYZ_to_xyY(RGB_to_XYZ(rgb_709, D65_WHITE, D65_WHITE,
                                    BT709_COLOURSPACE.RGB_to_XYZ_matrix))
    xy_image = tpg.get_chromaticity_image(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    cmf_xy = tpg._get_cmfs_xy()
    xlim = (min(0, xmin), max(0.8, xmax))
    ylim = (min(0, ymin), max(0.9, ymax))
    figsize_h = 8 * 1.0
    figsize_v = 9 * 1.0
    rate = 1.3
    # gamut の用意
    outer_gamut = np.array(tpg.get_primaries(cs.BT2020)[0])
    inner_gamut = np.array(tpg.get_primaries(cs.BT709)[0])
    outer_name = 'ITR-R BT.2020'
    inner_name = 'ITR-R BT.709'
    ax1 = pu.plot_1_graph(fontsize=20 * rate,
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
                          minor_xtick_num=2, minor_ytick_num=2)
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=3.5*rate, label=None)
    ax1.plot((cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
             '-k', lw=3.5*rate, label=None)
    ax1.plot(inner_gamut[:, 0], inner_gamut[:, 1],
             c=UNIVERSAL_COLOR_LIST[0], label=inner_name, lw=2.75*rate)
    ax1.plot(outer_gamut[:, 0], outer_gamut[:, 1],
             c=UNIVERSAL_COLOR_LIST[3], label=outer_name, lw=2.75*rate)
    ax1.plot(tpg.D65_WHITE[0], tpg.D65_WHITE[1], 'x', c='k',
             lw=2.75*rate, label='D65', ms=10*rate, mew=2.75*rate)
    ax1.plot(src_xyY[..., 0], src_xyY[..., 1], ls='', marker='o',
             c='#000000', ms=5.5*rate, label="Before")
    ax1.plot(dst_xyY[..., 0], dst_xyY[..., 1], ls='', marker='+',
             c='#808080', ms=10*rate, mew=2.0*rate,
             label="After")
    # annotation
    arrowprops = dict(
        facecolor='#333333', shrink=0.0, headwidth=6, headlength=8,
        width=1)
    src_xyY = src_xyY.reshape((18, 8, 3))
    dst_xyY = dst_xyY.reshape((18, 8, 3))
    for h_idx in range(18):
        for s_idx in range(8):
            if s_idx < 3:
                continue
            ed_pos = (dst_xyY[h_idx][s_idx][0],
                      dst_xyY[h_idx][s_idx][1])
            st_pos = (src_xyY[h_idx][s_idx][0],
                      src_xyY[h_idx][s_idx][1])
            ax1.annotate("", xy=ed_pos, xytext=st_pos, xycoords='data',
                         textcoords='data', ha='left', va='bottom',
                         arrowprops=arrowprops)

    ax1.imshow(xy_image, extent=(xmin, xmax, ymin, ymax))
    plt.legend(loc='upper right', fontsize=14 * rate)
    png_file_name = "./blog_img/xyY_plot_simple_mtx_conv.png"
    plt.savefig(png_file_name, bbox_inches='tight')
    plt.show()


def plot_apply_eetf_code_value():
    """
    推測したEETFを実際に適用して合っているか確認する。
    """
    st_h = 115
    st_v = 563
    ed_h = st_h + 2048

    img_youtube = cv2.imread("./img/youtubed_sdr_tp_all.png")
    img_src = cv2.imread("./img/youtubed_sdr_src.tiff",
                         cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[:, :, ::-1]
    x = np.arange(2048) / 2.0 / 1023.0
    x_src = img_src[st_v, st_h:ed_h, 1] / 65535.0
    y_ref = tf.oetf_from_luminance(
        ((img_youtube[st_v, st_h:ed_h, 1] / 255.0) ** 2.4) * 100, tf.ST2084)
    y_formula = youtube_tonemapping(x_src)
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Tone Mapping Characteristics",
        graph_title_size=None,
        xlabel="ST2084 Code Value",
        ylabel="Luminance [cd/m2]",
        axis_label_size=None,
        legend_size=17,
        xtick_size=19,
        ytick_size=19,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        linewidth=8,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(x, y_ref, label='ref')
    ax1.plot(x, y_formula, label='formula')
    plt.legend(loc='upper left')
    plt.savefig(
        "./blog_img/check_estimated_formula_code_value.png",
        bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_apply_eetf_luminance():
    """
    推測したEETFを実際に適用して合っているか確認する。
    """
    st_h = 115
    st_v = 563
    ed_h = st_h + 2048

    img_youtube = cv2.imread("./img/youtubed_sdr_tp_all.png")
    img_src = cv2.imread("./img/youtubed_sdr_src.tiff",
                         cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[:, :, ::-1]
    x = np.arange(2048) / 2.0
    x_src = img_src[st_v, st_h:ed_h, 1] / 65535.0
    y_ref = img_youtube[st_v, st_h:ed_h, 1] / 255.0
    y_eetf = youtube_tonemapping(x_src)
    y_formula = tf.oetf_from_luminance(
        tf.eotf_to_luminance(y_eetf, tf.ST2084), tf.GAMMA24)
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Tone Mapping Characteristics",
        graph_title_size=None,
        xlabel="Luminance [cd/m2]",
        ylabel="Luminance [cd/m2]",
        axis_label_size=None,
        legend_size=17,
        xtick_size=19,
        ytick_size=19,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        linewidth=8,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(x, y_ref, label='ref')
    ax1.plot(x, y_formula, label='formula')
    plt.legend(loc='upper left')
    plt.savefig(
        "./blog_img/check_estimated_formula_luinance.png",
        bbox_inches='tight', pad_inches=0.1)
    plt.show()


def main_func():
    # correct_pq_exr_gain()
    # check_after_3dlut_exr()
    analyze_eetf()
    # analyze_gamut_mapping()
    # x = np.linspace(0, 1, 1024)
    # plt.plot(x, eotf_BT1886(x, L_B=0.0), label="zero")
    # plt.plot(x, eotf_BT1886(x, L_B=1/1000), label="1/1000")
    # plt.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
