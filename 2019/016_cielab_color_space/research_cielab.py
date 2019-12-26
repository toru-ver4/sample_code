# -*- coding: utf-8 -*-
"""
CIELAB色空間の基礎調査
=====================

* XYZ色空間と Lab色空間の順変換・逆変換の数式を確認
* CLELAB a*b* plane (以後は a*b* plane と略す) のプロット(L を 0～100 まで 0.1 step で)
* CIELAB C*L* plane (以後は C*L* plane と略す) のプロット(h を 0～360 まで 0.5 step で)

"""

# import standard libraries
import os
import time
import ctypes

# import third-party libraries
# import matplotlib as mpl
# mpl.use('Agg')
import numpy as np
from sympy import symbols, plotting, sin, cos, lambdify
from sympy.solvers import solve
from scipy import linalg
from colour.models import BT2020_COLOURSPACE, BT709_COLOURSPACE
from colour import xy_to_XYZ, read_image, write_image, Lab_to_XYZ, XYZ_to_RGB
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, Array

# import my libraries
import color_space as cs
import plot_utility as pu

# definition
D65_X = 95.04
D65_Y = 100.0
D65_Z = 108.89
D65_WHITE = [D65_X, D65_Y, D65_Z]
SIGMA = 6/29

IJK_LIST = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]]

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

# global variables
l_sample_num = 4
h_sample_num = 4
shared_array = Array(
    typecode_or_type=ctypes.c_float,
    size_or_initializer=l_sample_num*h_sample_num)


def get_ty(l):
    """
    l, c, h は Sympy の Symbol
    """
    return (l + 16) / 116


def get_tx(l, c, h):
    """
    l, c, h は Sympy の Symbol
    """
    return get_ty(l) + (c * cos(h))/500


def get_tz(l, c, h):
    """
    l, c, h は Sympy の Symbol
    """
    return get_ty(l) - (c * sin(h))/200


def get_large_x(l, c, h, ii):
    """
    l, c, h は Sympy の Symbol.
    ii==0 は t <= sigma, ii==1 は t > sigma のパターン。
    """
    if ii == 0:
        ret_val = D65_X / 100 * 3 * (SIGMA ** 2) * (get_tx(l, c, h) - 4 / 29)
    else:
        ret_val = D65_X / 100 * (get_tx(l, c, h) ** 3)

    return ret_val


def get_large_y(l, c, h, jj):
    """
    l, c, h は Sympy の Symbol.
    jj==0 は t <= sigma, jj==1 は t > sigma のパターン。
    """
    if jj == 0:
        ret_val = D65_Y / 100 * 3 * (SIGMA ** 2) * (get_ty(l) - 4 / 29)
    else:
        ret_val = D65_Y / 100 * (get_ty(l) ** 3)

    return ret_val


def get_large_z(l, c, h, kk):
    """
    l, c, h は Sympy の Symbol.
    kk==0 は t <= sigma, kk==1 は t > sigma のパターン。
    """
    if kk == 0:
        ret_val = D65_Z / 100 * 3 * (SIGMA ** 2) * (get_tz(l, c, h) - 4 / 29)
    else:
        ret_val = D65_Z / 100 * (get_tz(l, c, h) ** 3)

    return ret_val


def get_xyz_to_rgb_matrix(primaries=cs.REC2020_xy):
    rgb_to_xyz_matrix = cs.calc_rgb_to_xyz_matrix(
        gamut_xy=primaries, white_large_xyz=D65_WHITE)
    xyz_to_rgb_matrix = linalg.inv(rgb_to_xyz_matrix)
    return xyz_to_rgb_matrix


def lab_to_rgb_expr(l, c, h):
    mtx = get_xyz_to_rgb_matrix(primaries=cs.REC2020_xy)
    ret_list = []
    for ijk in IJK_LIST:
        r = mtx[0][0] * get_large_x(l, c, h, ijk[0]) + mtx[0][1] * get_large_y(l, c, h, ijk[1]) + mtx[0][2] * get_large_z(l, c, h, ijk[2])
        g = mtx[1][0] * get_large_x(l, c, h, ijk[0]) + mtx[1][1] * get_large_y(l, c, h, ijk[1]) + mtx[1][2] * get_large_z(l, c, h, ijk[2])
        b = mtx[2][0] * get_large_x(l, c, h, ijk[0]) + mtx[2][1] * get_large_y(l, c, h, ijk[1]) + mtx[2][2] * get_large_z(l, c, h, ijk[2])
        ret_list.append([r, g, b])

    return ret_list


def visualize_formula():
    """
    C* と RGB値のグラフをプロット
    """
    l, c, h = symbols('l, c, h', real=True)
    rgb_exprs = lab_to_rgb_expr(l, c, h)
    l_vals = np.linspace(0, 100, l_sample_num)
    h_vals = np.linspace(0, 2*np.pi, h_sample_num)
    for l_idx, l_val in enumerate(l_vals):
        args = []
        for h_idx, h_val in enumerate(h_vals):
            args.append([l_val, l_idx, h_val, h_idx, rgb_exprs, l, c, h])
            # plot_formula_for_specific_lstar(
            #     l_val, l_idx, h_val, h_idx, rgb_exprs, l, c, h)
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_visualization_formula, args)


def thread_wrapper_visualization_formula(args):
    plot_formula_for_specific_lstar(*args)


def plot_formula_for_specific_lstar(
        l_val, l_idx, h_val, h_idx, rgb_exprs, l, c, h):
    print(l_idx, h_idx)
    x = np.linspace(-250, 250, 1024)

    for ii in range(len(IJK_LIST)):

        for jj in range(3):
            # l_val, h_val 代入
            rgb_exprs[ii][jj] = rgb_exprs[ii][jj].subs({l: l_val, h: h_val})
            # lambdify 実行
            rgb_exprs[ii][jj] = lambdify(c, rgb_exprs[ii][jj], 'numpy')
            # プロット対象のY軸データ作成
            rgb_exprs[ii][jj] = rgb_exprs[ii][jj](x)

    # 1次元になっちゃうやつへの対処
    for ii in range(len(IJK_LIST)):
        for jj in range(3):
            if not isinstance(rgb_exprs[ii][jj], np.ndarray):
                rgb_exprs[ii][jj] = np.ones_like(x) * rgb_exprs[ii][jj]

    # upper_rgb と lower_rgb を合成
    xyz_t = [get_tx(l, c, h), get_ty(l), get_tz(l, c, h)]
    xyz_t = [xyz_t[idx].subs({l: l_val, h: h_val}) for idx in range(3)]
    xyz_t = [lambdify(c, xyz_t[idx], 'numpy') for idx in range(3)]
    xyz_t = [xyz_t[idx](x) for idx in range(3)]
    for idx in range(3):
        if not isinstance(xyz_t[idx], np.ndarray):
            xyz_t[idx] = np.ones_like(x) * xyz_t[idx]

    rgb = [np.zeros_like(rgb_exprs[0][idx]) for idx in range(3)]
    for ii in range(len(IJK_LIST)):
        for jj in range(3):
            r_idx = (xyz_t[jj] <= SIGMA) if IJK_LIST[ii][jj] else (xyz_t[jj] > SIGMA)
            g_idx = (xyz_t[jj] <= SIGMA) if IJK_LIST[ii][jj] else (xyz_t[jj] > SIGMA)
            b_idx = (xyz_t[jj] <= SIGMA) if IJK_LIST[ii][jj] else (xyz_t[jj] > SIGMA)
            idx = (r_idx & g_idx) & b_idx
            rgb[jj][idx] = rgb_exprs[ii][jj][idx]

    graph_name_0 = "./formula_seq/L0_{:03d}_{:04d}.png".format(l_idx, h_idx)
    graph_name_1 = "./formula_seq/L1_{:03d}_{:04d}.png".format(l_idx, h_idx)
    graph_name_2 = "./formula_seq/L_{:03d}_{:04d}.png".format(l_idx, h_idx)
    title_str = "L*={:.02f}_H={:.01f}°".format(
        100 * l_idx / (l_sample_num - 1), 360 * h_idx / (h_sample_num - 1))
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
        graph_title=title_str,
        graph_title_size=None,
        xlabel="C*", ylabel="RGB Value",
        axis_label_size=None,
        legend_size=17,
        xlim=(-50, 250),
        ylim=(-0.5, 0.5),
        xtick=[25 * x - 50 for x in range(13)],
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(x, rgb[0], 'r-', label="R")
    ax1.plot(x, rgb[1], 'g-', label="G")
    ax1.plot(x, rgb[2], 'b-', label="B")
    ax1.plot(x, xyz_t[0] > SIGMA, 'r-', alpha=0.3, label="are")
    ax1.plot(x, xyz_t[1] > SIGMA, 'g-', alpha=0.3, label="are")
    ax1.plot(x, xyz_t[2] > SIGMA, 'b-', alpha=0.3, label="are")
    plt.legend(loc='upper left')
    plt.savefig(graph_name_0, bbox_inches='tight', pad_inches=0.1)
    # plt.show()

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
        graph_title=title_str,
        graph_title_size=None,
        xlabel="C*", ylabel="RGB Value",
        axis_label_size=None,
        legend_size=17,
        xlim=(-50, 250),
        ylim=(0.5, 1.5),
        xtick=[25 * x - 50 for x in range(13)],
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(x, rgb[0], 'r-', label="R")
    ax1.plot(x, rgb[1], 'g-', label="G")
    ax1.plot(x, rgb[2], 'b-', label="B")
    ax1.plot(x, xyz_t[0] > SIGMA, 'r-', alpha=0.3, label="are")
    ax1.plot(x, xyz_t[1] > SIGMA, 'g-', alpha=0.3, label="are")
    ax1.plot(x, xyz_t[2] > SIGMA, 'b-', alpha=0.3, label="are")
    plt.legend(loc='upper left')
    plt.savefig(graph_name_1, bbox_inches='tight', pad_inches=0.1)
    # plt.show()

    img_0 = read_image(graph_name_0)
    img_1 = read_image(graph_name_1)
    img = np.hstack((img_0, img_1))
    write_image(img, graph_name_2)
    os.remove(graph_name_0)
    os.remove(graph_name_1)


def experimental_functions():
    visualize_formula()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    experimental_functions()
