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
from colour import xyY_to_XYZ, XYZ_to_RGB
from colour.models import BT709_COLOURSPACE
from sympy import symbols, solve
from multiprocessing import Pool, cpu_count, Array
from mpl_toolkits.mplot3d import Axes3D
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
h_sample = 128
shared_array = Array(
    typecode_or_type=ctypes.c_float,
    size_or_initializer=y_sample*h_sample)


def calc_xyY(large_y, hue):
    sample_num = 1024
    c = np.linspace(0, 1.0, sample_num)
    x = c * np.cos(hue)
    y = c * np.sin(hue)
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
    rgb = np.clip(rgb, 0.0, 1.0) ** 1/2.4
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
    ax1.plot(x, y, 'k-')
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
            plot_xy_plane(y_idx, chroma[y_idx])
        pool.map(plot_xy_plane_thread, args)


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


def main_func():
    # check_large_xyz_boundary(large_y=0.7, hue=45/360*2*np.pi)
    # calc_point_ab_rgb()
    # solve_chroma(0.1, 45/360*2*np.pi)
    # chroma = calc_all_chroma()
    # np.save("cie1931_chroma.npy", chroma)
    chroma = np.load("cie1931_chroma.npy")
    plot_xy_pane_all(chroma)
    # plot_xyY_color_volume(chroma)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
