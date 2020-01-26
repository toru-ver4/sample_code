# -*- coding: utf-8 -*-
"""
XYZ空間やL*a*b*空間の境界線の確認
======================================

XYZ空間やL*a*b*空間の境界線を確認する

"""

# import standard libraries
import os
import ctypes

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from colour import xyY_to_XYZ, XYZ_to_RGB, read_image, write_image, LUT3D,\
    RGB_to_XYZ, XYZ_to_xyY
from colour.models import BT709_COLOURSPACE
from sympy import symbols, solve
from multiprocessing import Pool, cpu_count, Array
from mpl_toolkits.mplot3d import Axes3D
import test_pattern_generator2 as tpg
import cv2
# import matplotlib as mpl
# mpl.use('Agg')

# import my libraries
import plot_utility as pu
import color_space as cs
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

# global variables
y_sample = 64
h_sample = 1024
shared_array = Array(
    typecode_or_type=ctypes.c_float,
    size_or_initializer=y_sample*h_sample)


def calc_xyY(large_y, hue):
    sample_num = 1024
    c = np.linspace(-0.2, 1.0, sample_num)
    x = c * np.cos(hue) + cs.D65[0]
    y = c * np.sin(hue) + cs.D65[1]
    large_y2 = np.ones(sample_num) * large_y
    xyY = np.dstack((x, y, large_y2))

    return xyY, x, y


def plot_rgb_around_large_xyz_boundary(rgb, x, y, large_y, hue):
    title_str =\
        "Y={:.02f}, Angle={:.01f}°".format(large_y, hue * 360 / (2 * np.pi))
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title=title_str,
        graph_title_size=None,
        xlabel="x",
        ylabel="RGB Value",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=(-0.1, 1.1),
        xtick=[0.0, 0.185, 0.3, 0.467, 0.6],
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(x, rgb[..., 0].flatten(), '-r', label="Red")
    ax1.plot(x, rgb[..., 1].flatten(), '-g', label="Green")
    ax1.plot(x, rgb[..., 2].flatten(), '-b', label="Blue")
    plt.legend(loc='upper right')
    fname = "./blog_img/" + title_str + ".png"
    fname = fname.replace("=", "_")
    fname = fname.replace("°", "")
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_line_with_chromaticity_diagram(
        x, y, large_y, hue,
        rate=480/755.0*2, xmin=0.0, xmax=0.8, ymin=0.0, ymax=0.9):
    """
    直線とxy色度図をプロットしてみる
    """
    title_str =\
        "Y={:.02f}, Angle={:.01f}°".format(large_y, hue * 360 / (2 * np.pi))

    # プロット用データ準備
    # ---------------------------------
    xy_image = tpg.get_chromaticity_image(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    cmf_xy = tpg._get_cmfs_xy()

    bt709_gamut, _ = tpg.get_primaries(name=cs.BT709)
    xlim = (min(0, xmin), max(0.8, xmax))
    ylim = (min(0, ymin), max(0.9, ymax))

    ax1 = pu.plot_1_graph(fontsize=20 * rate,
                          figsize=((xmax - xmin) * 10 * rate,
                                   (ymax - ymin) * 10 * rate),
                          graph_title="CIE1931 Chromaticity Diagram",
                          graph_title_size=None,
                          xlabel='x', ylabel='y',
                          axis_label_size=None,
                          legend_size=18 * rate,
                          xlim=xlim, ylim=ylim,
                          xtick=[x * 0.1 + xmin for x in
                                 range(int((xlim[1] - xlim[0])/0.1) + 1)],
                          ytick=[x * 0.1 + ymin for x in
                                 range(int((ylim[1] - ylim[0])/0.1) + 1)],
                          xtick_size=17 * rate,
                          ytick_size=17 * rate,
                          linewidth=4 * rate,
                          minor_xtick_num=2,
                          minor_ytick_num=2)
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=3.5*rate, label=None)
    ax1.plot((cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
             '-k', lw=3.5*rate, label=None)
    ax1.plot(bt709_gamut[:, 0], bt709_gamut[:, 1],
             c=tpg.UNIVERSAL_COLOR_LIST[2], label="BT.709", lw=2.75*rate)
    ax1.plot(x, y, 'k-', label="Line", lw=2.75*rate)

    ax1.imshow(xy_image, extent=(xmin, xmax, ymin, ymax))
    plt.legend(loc='upper right')
    fname = "./blog_img/chroma_" + title_str + ".png"
    fname = fname.replace("=", "_")
    fname = fname.replace("°", "")
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_chromaticity_diagram(
        rate=480/755.0*2, xmin=0.0, xmax=0.8, ymin=0.0, ymax=0.9):
    """
    直線とxy色度図をプロットしてみる
    """
    title_str = "Chromaticity Diagram"

    # プロット用データ準備
    # ---------------------------------
    xy_image = tpg.get_chromaticity_image(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    cmf_xy = tpg._get_cmfs_xy()

    bt709_gamut, _ = tpg.get_primaries(name=cs.BT709)
    xlim = (min(0, xmin), max(0.8, xmax))
    ylim = (min(0, ymin), max(0.9, ymax))

    ax1 = pu.plot_1_graph(fontsize=20 * rate,
                          figsize=((xmax - xmin) * 10 * rate,
                                   (ymax - ymin) * 10 * rate),
                          graph_title="CIE1931 Chromaticity Diagram",
                          graph_title_size=None,
                          xlabel='x', ylabel='y',
                          axis_label_size=None,
                          legend_size=18 * rate,
                          xlim=xlim, ylim=ylim,
                          xtick=[x * 0.1 + xmin for x in
                                 range(int((xlim[1] - xlim[0])/0.1) + 1)],
                          ytick=[x * 0.1 + ymin for x in
                                 range(int((ylim[1] - ylim[0])/0.1) + 1)],
                          xtick_size=17 * rate,
                          ytick_size=17 * rate,
                          linewidth=4 * rate,
                          minor_xtick_num=2,
                          minor_ytick_num=2)
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=3.5*rate, label=None)
    ax1.plot((cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
             '-k', lw=3.5*rate, label=None)
    ax1.plot(bt709_gamut[:, 0], bt709_gamut[:, 1],
             c=tpg.UNIVERSAL_COLOR_LIST[2], label="BT.709", lw=2.75*rate)
    ax1.plot([0.35], [0.2], 'o', c="#000000", ms=12, label="A")
    ax1.plot([0.45], [0.2], 'o', c="#808080", ms=12, label="B")
    arrowprops = dict(
        facecolor='#A0A0A0', shrink=0.0, headwidth=8, headlength=10,
        width=2)
    ax1.annotate("A", xy=[0.35, 0.2], xytext=[0.45, 0.3], xycoords='data',
                 textcoords='data', ha='left', va='bottom',
                 arrowprops=arrowprops)
    ax1.annotate("B", xy=[0.45, 0.2], xytext=[0.55, 0.3], xycoords='data',
                 textcoords='data', ha='left', va='bottom',
                 arrowprops=arrowprops)

    ax1.imshow(xy_image, extent=(xmin, xmax, ymin, ymax))
    plt.legend(loc='upper right')
    fname = "./blog_img/" + title_str + ".png"
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def apply_matrix(src, mtx):
    """
    src: [3]
    mtx: [3][3]
    """
    a = src[0] * mtx[0][0] + src[1] * mtx[0][1] + src[2] * mtx[0][2]
    b = src[0] * mtx[1][0] + src[1] * mtx[1][1] + src[2] * mtx[1][2]
    c = src[0] * mtx[2][0] + src[1] * mtx[2][1] + src[2] * mtx[2][2]

    return a, b, c


def xyY_to_XYZ_local(x, y, large_y):
    """
    # 概要
    xyYからXYZを計算する

    # 入力データ
    numpy形式。shape = (1, N, 3)
    """
    z = 1 - x - y
    yx = y / x
    yz = y / z
    large_x = large_y / yx
    large_z = large_y / yz

    return large_x, large_y, large_z


def get_rgb_formula(large_y, hue):
    c = symbols('c', real=True)
    x = c * np.cos(hue)
    y = c * np.sin(hue)
    large_x, large_y, large_z = xyY_to_XYZ_local(x, y, large_y)
    mtx = BT709_COLOURSPACE.XYZ_to_RGB_matrix
    r, g, b = apply_matrix([large_x, large_y, large_z], mtx)

    result = []

    result.extend(solve(r + 0))
    result.extend(solve(g + 0))
    result.extend(solve(b + 0))
    result.extend(solve(r - 1))
    result.extend(solve(g - 1))
    result.extend(solve(b - 1))

    result = np.array(result)
    print(result[result > 0] * np.cos(hue))


def calc_point_ab_rgb():
    ab = np.array([[[0.35, 0.2, 0.05], [0.45, 0.2, 0.05]]])
    rgb = XYZ_to_RGB(
        xyY_to_XYZ(ab), cs.D65, cs.D65, BT709_COLOURSPACE.XYZ_to_RGB_matrix)
    print(rgb)


def calc_point_xyY_rgb():
    D65 = [0.3127, 0.329]
    a = np.array([0.35, 0.2, 0.05])
    b = np.array([0.45, 0.2, 0.05])
    a_rgb = XYZ_to_RGB(
        xyY_to_XYZ(a), D65, D65, BT709_COLOURSPACE.XYZ_to_RGB_matrix)
    b_rgb = XYZ_to_RGB(
        xyY_to_XYZ(b), D65, D65, BT709_COLOURSPACE.XYZ_to_RGB_matrix)
    print("Point A = {:.3f}, {:.3f}, {:.3f}".format(a_rgb[0], a_rgb[1], a_rgb[2]))
    print("Point B = {:.3f}, {:.3f}, {:.3f}".format(b_rgb[0], b_rgb[1], b_rgb[2]))


def check_large_xyz_boundary(large_y, hue):
    """
    XYZ空間の境界線でのRGB値の変化を確認する
    """
    xyY, x, y = calc_xyY(large_y=large_y, hue=hue)
    rgb = XYZ_to_RGB(
        xyY_to_XYZ(xyY), cs.D65, cs.D65, BT709_COLOURSPACE.XYZ_to_RGB_matrix)
    plot_rgb_around_large_xyz_boundary(rgb, x, y, large_y=large_y, hue=hue)
    plot_line_with_chromaticity_diagram(x, y, large_y, hue)
    get_rgb_formula(large_y, hue)


def solve_chroma(large_y, hue):
    """
    large_y, hue から xyY を計算し、
    更に XYZ to RGB 変換して RGB値の境界の Chroma を出す。
    """
    c = symbols('c', real=True)
    x = c * np.cos(hue) + cs.D65[0]
    y = c * np.sin(hue) + cs.D65[1]
    large_xyz = xyY_to_XYZ_local(x, y, large_y)
    mtx = BT709_COLOURSPACE.XYZ_to_RGB_matrix
    r, g, b = apply_matrix([large_xyz[0], large_xyz[1], large_xyz[2]], mtx)
    chroma = []
    chroma.extend(solve(r + 0))
    chroma.extend(solve(g + 0))
    chroma.extend(solve(b + 0))
    chroma.extend(solve(r - 1))
    chroma.extend(solve(g - 1))
    chroma.extend(solve(b - 1))

    chroma = np.array(chroma)
    if chroma != []:
        chroma = np.min(chroma[chroma >= 0])
    else:
        chroma = 0.0

    # result_x = chroma * np.cos(hue) + cs.D65[0]
    # result_y = chroma * np.sin(hue) + cs.D65[1]

    return chroma


def solve_chroma_thread(args):
    chroma = solve_chroma(args[1], args[2])
    print("y_idx={}, h_idx={}".format(args[1], args[2]))
    shared_array[args[0]] = chroma


def calc_all_chroma():
    large_y = np.linspace(0.0, 1.0, y_sample)
    hue = np.linspace(0.0, 2*np.pi, h_sample)
    for y_idx, y in enumerate(large_y):
        with Pool(cpu_count()) as pool:
            args = []
            for h_idx, h in enumerate(hue):
                idx = h_sample * y_idx + h_idx
                args.append([idx, y, h])
            pool.map(solve_chroma_thread, args)

    chroma = np.array(shared_array[:]).reshape((y_sample, h_sample))
    return chroma


def plot_xy_plane(y_idx, chroma):
    graph_name = "./xy_plane_seq/no_{:04d}.png".format(y_idx)
    rad = np.linspace(0, 2 * np.pi, h_sample)
    large_y = 1.0 * y_idx / (y_sample - 1)
    if y_idx < (y_sample - 1):
        x = chroma * np.cos(rad) + cs.D65[0]
        y = chroma * np.sin(rad) + cs.D65[1]
    else:
        x = np.ones_like(chroma) * cs.D65[0]
        y = np.ones_like(chroma) * cs.D65[1]
    ly = np.ones_like(y) * large_y
    large_xyz = xyY_to_XYZ(np.dstack((x, y, ly)))
    rgb = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65, BT709_COLOURSPACE.XYZ_to_RGB_matrix)
    rgb = rgb.reshape((rgb.shape[1], rgb.shape[2]))
    rgb = np.clip(rgb, 0.0, 1.0) ** (1/2.4)
    cmf_xy = tpg._get_cmfs_xy()

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(8, 9),
        graph_title="Y={:.03f} xy plane".format(large_y),
        graph_title_size=None,
        xlabel="x", ylabel="y",
        axis_label_size=None,
        legend_size=17,
        xlim=(0.0, 0.8),
        ylim=(0.0, 0.9),
        xtick=[x * 0.1 for x in range(9)],
        ytick=[x * 0.1 for x in range(10)],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.patch.set_facecolor("#B0B0B0")
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=3, label=None)
    ax1.plot((cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
             '-k', lw=3, label=None)

    # ax1.plot(x, y, 'k-')
    ax1.scatter(x, y, c=rgb)
    # plt.legend(loc='upper left')
    plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    print("plot y_idx={}".format(y_idx))
    # plt.show()


def plot_xy_plane_thread(args):
    plot_xy_plane(*args)


def plot_xy_pane_all(chroma):
    args = []
    with Pool(cpu_count()) as pool:
        for y_idx in range(y_sample):
            args.append([y_idx, chroma[y_idx]])
            # plot_xy_plane(y_idx, chroma[y_idx])
        pool.map(plot_xy_plane_thread, args)


def plot_xyY_plane_seq(chroma):
    args = []
    with Pool(cpu_count()) as pool:
        for y_idx in range(y_sample):
            args.append([y_idx, chroma])
            # plot_xy_plane(y_idx, chroma[y_idx])
        pool.map(plot_xyY_plane_seq_thread, args)


def plot_xyY_plane_seq_thread(args):
    plot_xyY_color_volume_seq(*args)


def plot_xyY_color_volume(chroma):
    rad = np.linspace(0, 2 * np.pi, h_sample)
    large_y = np.linspace(0, 1, y_sample).reshape((y_sample, 1))
    chroma[-1, :] = 0.0
    x = chroma * np.cos(rad) + cs.D65[0]
    y = chroma * np.sin(rad) + cs.D65[1]
    ly = np.ones_like(y) * large_y
    large_xyz = xyY_to_XYZ(np.dstack((x, y, ly)))
    rgb = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65, BT709_COLOURSPACE.XYZ_to_RGB_matrix)
    rgb = np.clip(rgb, 0.0, 1.0) ** (1/2.4)
    rgb = rgb.reshape((y_sample * h_sample, 3))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Y")
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(0.0, 0.9)
    ax.view_init(elev=20, azim=-120)
    ax.plot(x.flatten(), y.flatten(), ly.flatten())
    plt.savefig("./blog_img/xyY_Color_Volume_wire.png",
                bbox_inches='tight', pad_inches=0.1)
    ax.scatter(x.flatten(), y.flatten(), ly.flatten(),
               marker='o', linestyle='-', c=rgb)
    plt.savefig("./blog_img/xyY_Color_Volume_color.png",
                bbox_inches='tight', pad_inches=0.1)


def plot_xyY_color_volume_seq(y_idx, chroma):
    graph_name_0 = "./xy_plane_seq/no_{:04d}.png".format(y_idx)
    graph_name_1 = "./xyY_color_volume_seq/no_{:04d}.png".format(y_idx)

    rad = np.linspace(0, 2 * np.pi, h_sample)
    large_y = np.linspace(0, 1, y_sample).reshape((y_sample, 1))
    large_y = large_y[:y_idx+1]
    chroma[-1, :] = 0.0
    x = chroma[:y_idx+1] * np.cos(rad) + cs.D65[0]
    y = chroma[:y_idx+1] * np.sin(rad) + cs.D65[1]
    ly = np.ones_like(y) * large_y
    large_xyz = xyY_to_XYZ(np.dstack((x, y, ly)))
    print(large_xyz.shape)
    rgb = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65, BT709_COLOURSPACE.XYZ_to_RGB_matrix)
    rgb = np.clip(rgb, 0.0, 1.0) ** (1/2.4)
    rgb = rgb.reshape(((y_idx + 1) * h_sample, 3))

    fig = plt.figure(figsize=(9, 9))
    ax = Axes3D(fig)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Y")
    ax.set_title("Y={:.03f} xyY Color Volume".format(large_y[-1, 0]),
                 fontsize=18)
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(0.0, 0.9)
    ax.set_zlim(0.0, 1.1)
    ax.view_init(elev=20, azim=-120)
    ax.scatter(x.flatten(), y.flatten(), ly.flatten(),
               marker='o', c=rgb, zorder=1)
    plt.savefig(graph_name_1, bbox_inches='tight', pad_inches=0.1)

    resize_and_hstack(graph_name_0, graph_name_1)


def resize_and_hstack(fname1, fname2):
    print(fname1, fname2)
    img_0 = read_image(fname1)
    img_1 = read_image(fname2)

    if img_0.shape[0] > img_1.shape[0]:
        static_img = img_0
        resize_img = img_1
    else:
        static_img = img_1
        resize_img = img_0

    rate = static_img.shape[0] / resize_img.shape[0]
    dst_size = (int(resize_img.shape[1] * rate),
                int(resize_img.shape[0] * rate))
    resize_img = cv2.resize(resize_img, dst_size)
    img = np.hstack((resize_img, static_img))
    write_image(img, fname2)


def make_rgb_figure(y_idx=y_sample//2):
    large_y = np.linspace(0, 1, y_sample)[y_idx]
    hue_array = np.linspace(0.0, 2*np.pi, h_sample)
    chroma = np.load("cie1931_chroma.npy")

    args = []
    for h_idx, hue in enumerate(hue_array):
        xyY, x, y = calc_xyY(large_y=large_y, hue=hue)
        rgb = XYZ_to_RGB(xyY_to_XYZ(xyY), cs.D65, cs.D65,
                         BT709_COLOURSPACE.XYZ_to_RGB_matrix)
        args.append([rgb, large_y, hue, chroma[y_idx][h_idx], h_idx])
        # plot_rgb_formula(rgb, large_y=large_y, hue=hue)
    with Pool(cpu_count()) as pool:
        pool.map(plot_rgb_formula_thread, args)


def plot_rgb_formula_thread(args):
    plot_rgb_formula(*args)


def plot_rgb_formula(rgb, large_y, hue, chroma, h_idx):
    sample_num = 1024
    c = np.linspace(-0.2, 1.0, sample_num)
    title_str =\
        "Y={:.02f}, H={:.01f}°".format(large_y, hue * 360 / (2 * np.pi))
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(8, 8),
        graph_title=title_str,
        graph_title_size=None,
        xlabel="C",
        ylabel="RGB Value",
        axis_label_size=None,
        legend_size=17,
        xlim=(-0.2, 0.8),
        ylim=(-0.2, 1.2),
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    chroma_x = np.array([chroma, chroma])
    chroma_y = np.array([-0.2, 1.2])
    ax1.plot(c, rgb[..., 0].flatten(), '-r', label="Red")
    ax1.plot(c, rgb[..., 1].flatten(), '-g', label="Green")
    ax1.plot(c, rgb[..., 2].flatten(), '-b', label="Blue")
    ax1.plot(chroma_x, chroma_y, '--k', label='gamut boundary')
    plt.legend(loc='upper right')
    fname_str = "./formula_seq/Y_{:.02f}_Angle_{:04d}.png"
    fname = fname_str.format(large_y, h_idx)
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
    # plt.show()


def plot_xy_plane_little_by_little_seq(y_idx=y_sample//2):
    hue_array = np.linspace(0.0, 2*np.pi, h_sample)
    chroma = np.load("cie1931_chroma.npy")

    args = []
    for h_idx, hue in enumerate(hue_array):
        args.append([y_idx, chroma[y_idx], h_idx])
        # plot_xy_plane_little_by_little(y_idx, chroma[y_idx], h_idx)
    with Pool(cpu_count()) as pool:
        pool.map(plot_xy_plane_little_by_little_thread, args)


def plot_xy_plane_little_by_little_thread(args):
    plot_xy_plane_little_by_little(*args)


def plot_xy_plane_little_by_little(y_idx, chroma, h_idx):
    graph_name = "./xy_plane_seq/no_{:04d}.png".format(h_idx)
    hue = h_idx / (h_sample - 1) * 360
    rad = np.linspace(0, 2 * np.pi, h_sample)
    large_y = 1.0 * y_idx / (y_sample - 1)
    if y_idx < (y_sample - 1):
        x = chroma * np.cos(rad) + cs.D65[0]
        y = chroma * np.sin(rad) + cs.D65[1]
    else:
        x = np.ones_like(chroma) * cs.D65[0]
        y = np.ones_like(chroma) * cs.D65[1]
    ly = np.ones_like(y) * large_y
    large_xyz = xyY_to_XYZ(np.dstack((x, y, ly)))
    rgb = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65, BT709_COLOURSPACE.XYZ_to_RGB_matrix)
    rgb = rgb.reshape((rgb.shape[1], rgb.shape[2]))
    rgb = np.clip(rgb, 0.0, 1.0) ** (1/2.4)
    cmf_xy = tpg._get_cmfs_xy()
    x = x[:h_idx+1]
    y = y[:h_idx+1]
    rgb = rgb[:h_idx+1]
    line_x = np.array([cs.D65[0], x[-1]])
    line_y = np.array([cs.D65[1], y[-1]])

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(8, 9),
        graph_title="Y={:.03f}, H={:.01f}° xy plane".format(large_y, hue),
        graph_title_size=None,
        xlabel="x", ylabel="y",
        axis_label_size=None,
        legend_size=17,
        xlim=(0.0, 0.8),
        ylim=(0.0, 0.9),
        xtick=[x * 0.1 for x in range(9)],
        ytick=[x * 0.1 for x in range(10)],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.patch.set_facecolor("#D0D0D0")
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=3, label=None)
    ax1.plot((cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
             '-k', lw=3, label=None)

    # ax1.plot(x, y, 'k-')
    ax1.plot(line_x, line_y, '-k', lw=2, zorder=1)
    ax1.scatter(x, y, c=rgb, zorder=2)
    # plt.legend(loc='upper left')
    plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    print("plot y_idx={}".format(y_idx))
    # plt.show()


def concat_graph():
    fname0_dir = "./formula_seq"
    fname1_dir = "./xy_plane_seq"
    fname2_dir = "./xy_plane_seq_concat"
    fname0_list = sorted(os.listdir(fname0_dir))
    fname1_list = sorted(os.listdir(fname1_dir))
    args = []
    for fname0, fname1 in zip(fname0_list, fname1_list):
        fname_0 = os.path.join(fname0_dir, fname0)
        fname_1 = os.path.join(fname1_dir, fname1)
        fname_2 = os.path.join(fname2_dir, fname0)
        args.append([fname_0, fname_1, fname_2])
        # resize_and_hstack(fname_0, fname_1)
    with Pool(cpu_count()) as pool:
        pool.map(concat_graph_thread, args)


def resize_and_hstack2(fname1, fname2, fname3):
    print(fname1, fname2)
    img_0 = read_image(fname1)
    img_1 = read_image(fname2)

    if img_0.shape[0] > img_1.shape[0]:
        static_img = img_0
        resize_img = img_1
    else:
        static_img = img_1
        resize_img = img_0

    rate = static_img.shape[0] / resize_img.shape[0]
    dst_size = (int(resize_img.shape[1] * rate),
                int(resize_img.shape[0] * rate))
    resize_img = cv2.resize(resize_img, dst_size)
    img = np.hstack((resize_img, static_img))
    write_image(img, fname3)


def concat_graph_thread(args):
    resize_and_hstack2(*args)


def verify_xy_gamut_boundary(sample=256):
    idx_list = np.array([8, 16, 24, 32, 48, 56])
    # idx_list = np.array([16])
    delta_large_y = 0.001
    y_list = idx_list / (y_sample - 1)
    chroma = np.load("cie1931_chroma.npy")

    rgb = LUT3D.linear_table(sample).reshape((1, sample ** 3, 3))
    xyz = RGB_to_XYZ(rgb, cs.D65, cs.D65, BT709_COLOURSPACE.RGB_to_XYZ_matrix)
    xyY = XYZ_to_xyY(xyz)
    for idx, y in enumerate(y_list):
        ok_idx = (y < xyY[:, :, 2]) & (xyY[:, :, 2] < (y + delta_large_y))
        verify_xy_gamut_boundary_plot(
            idx, y, xyY[ok_idx], rgb[ok_idx], chroma[idx_list[idx]])


def verify_xy_gamut_boundary_plot(y_idx, large_y, xyY, rgb, chroma):
    rad = np.linspace(0, 2 * np.pi, h_sample)
    x = chroma * np.cos(rad) + cs.D65[0]
    y = chroma * np.sin(rad) + cs.D65[1]
    cmf_xy = tpg._get_cmfs_xy()
    fname = "./blog_img/verify_y_{:.03f}.png".format(large_y)

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(8, 9),
        graph_title="Y={:.03f} xy plane".format(large_y),
        graph_title_size=None,
        xlabel="x", ylabel="y",
        axis_label_size=None,
        legend_size=17,
        xlim=(0.0, 0.8),
        ylim=(0.0, 0.9),
        xtick=[x * 0.1 for x in range(9)],
        ytick=[x * 0.1 for x in range(10)],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.patch.set_facecolor("#D0D0D0")
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=3, label=None)
    ax1.plot(x, y, c='k', lw=3, label='Gamut Boundary', alpha=0.5)
    ax1.plot((cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
             '-k', lw=3, label=None)
    ax1.scatter(xyY[..., 0], xyY[..., 1], c=rgb, s=2)
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
    # plt.show()


def main_func():
    # check_large_xyz_boundary(large_y=0.51, hue=90/360*2*np.pi)
    # calc_point_ab_rgb()
    # calc_point_xyY_rgb()
    # solve_chroma(0.1, 45/360*2*np.pi)
    # chroma = calc_all_chroma()
    # np.save("cie1931_chroma.npy", chroma)
    # chroma = np.load("cie1931_chroma.npy")
    # plot_xy_pane_all(chroma)
    # plot_xyY_plane_seq(chroma)
    # plot_xyY_color_volume(chroma)
    # make_rgb_figure(y_idx=y_sample//4)
    # plot_xy_plane_little_by_little_seq(y_idx=y_sample//4)
    # concat_graph()
    verify_xy_gamut_boundary()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
