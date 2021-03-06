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


def check_basic_trigonometricfunction():
    print(np.sin(np.pi * -4 / 4))
    print(np.sin(np.pi * -2 / 4))
    print(np.sin(np.pi * 2 / 4))
    print(np.sin(np.pi * 4 / 4))


def get_inv_f_upper():
    """
    t > sigma の f^-1 を返す
    """
    t = symbols('t')
    return t ** 3, t


def get_inv_f_lower():
    """
    t <= sigma の f^-1 を返す
    """
    t = symbols('t')
    sigma = SIGMA
    return 3 * (sigma ** 2) * (t - 4 / 29), t


def plot_inv_f():
    upper, t = get_inv_f_upper()
    plotting.plot(upper, (t, -1, 1))

    lower, t = get_inv_f_lower()
    plotting.plot(lower, (t, -1, 1))


def get_large_xyz_symbol(n, t, upper=True):
    """
    example
    -------
    c, l, h = symbols('c, l, h')
    xt = (l + 16) / 116 + (c * cos(h)) / 500
    yt = (l + 16) / 116
    zt = (l + 16) / 116 - (c * sin(h)) / 200

    x = get_large_xyz_symbol(n=D65_X, t=xt, upper=True)
    y = get_large_xyz_symbol(n=D65_Y, t=yt, upper=True)
    z = get_large_xyz_symbol(n=D65_Z, t=zt, upper=True)
    """
    func, u = get_inv_f_upper() if upper else get_inv_f_lower()
    return n / 100 * func.subs({u: t})


def apply_matrix(src, mtx):
    """
    src: [3]
    mtx: [3][3]
    """
    a = src[0] * mtx[0][0] + src[1] * mtx[0][1] + src[2] * mtx[0][2]
    b = src[0] * mtx[1][0] + src[1] * mtx[1][1] + src[2] * mtx[1][2]
    c = src[0] * mtx[2][0] + src[1] * mtx[2][1] + src[2] * mtx[2][2]

    return a, b, c


def get_xyz_to_rgb_matrix(primaries=cs.REC2020_xy):
    rgb_to_xyz_matrix = cs.calc_rgb_to_xyz_matrix(
        gamut_xy=primaries, white_large_xyz=D65_WHITE)
    xyz_to_rgb_matrix = linalg.inv(rgb_to_xyz_matrix)
    return xyz_to_rgb_matrix


def calc_chroma(h_sample=32):
    l_val = 50
    h = np.linspace(0, 1, h_sample) * 2 * np.pi

    chroma = []
    for h_val in h:
        chroma.append(lab_to_xyz_formla(l_val, h_val))
    return np.array(chroma)


def plot_ab_plane():
    h_sample = 32
    h = np.linspace(0, 1, h_sample) * 2 * np.pi
    chroma = calc_chroma(h_sample)
    a = chroma * np.cos(h)
    b = chroma * np.sin(h)

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Title",
        graph_title_size=None,
        xlabel="X Axis Label", ylabel="Y Axis Label",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(a, b, label="ab plane")
    plt.legend(loc='upper left')
    plt.show()


def get_xyz_t():
    """
    CIELAB to XYZ の逆関数の中の値を
    XYZ のぞれぞれについて求める。
    """
    c, l, h = symbols('c, l, h', real=True)
    xt = (l + 16) / 116 + (c * cos(h)) / 500
    yt = (l + 16) / 116
    zt = (l + 16) / 116 - (c * sin(h)) / 200
    xyz_t = [xt, yt, zt]

    return xyz_t, c, l, h


def lab_to_xyz_formla():
    """
    数式を取得
    """
    matrix = get_xyz_to_rgb_matrix(primaries=cs.REC2020_xy)

    # base formula
    xyz_t, c, l, h = get_xyz_t()

    # upper
    upper_xyzt = [
        get_large_xyz_symbol(n=D65_WHITE[idx], t=xyz_t[idx], upper=True)
        for idx in range(3)]
    upper_rgb = apply_matrix(upper_xyzt, matrix)

    # lower
    lower_xyzt = [
        get_large_xyz_symbol(n=D65_WHITE[idx], t=xyz_t[idx], upper=False)
        for idx in range(3)]
    lower_rgb = apply_matrix(lower_xyzt, matrix)

    # chroma = solve_chroma(upper_rgb, lower_rgb, xyz_t, l, h, l_val, h_val, c)

    return upper_rgb, lower_rgb, xyz_t, l, h, c


def solve_chroma():
    l_vals = np.linspace(0, 100, l_sample_num)
    h_vals = np.linspace(0, 2*np.pi, h_sample_num)
    upper_rgb, lower_rgb, xyz_t, l, h, c = lab_to_xyz_formla()
    args = []

    with Pool(cpu_count()) as pool:
        for l_idx, l_val in enumerate(l_vals):
            for h_idx, h_val in enumerate(h_vals):
                idx = h_sample_num * l_idx + h_idx
                args.append(
                    [l_val, h_val, idx, upper_rgb, lower_rgb, xyz_t,
                     l, h, c])
        pool.map(thread_wrapper, args)


def thread_wrapper(args):
    return solve_chroma_thread(*args)


def solve_chroma_thread(
        l_val, h_val, idx, upper_rgb, lower_rgb, xyz_t,
        l, h, c):
    result = solve_chroma_sub(
        upper_rgb, lower_rgb, xyz_t, l, h, l_val, h_val, c)
    shared_array[idx] = result


def solve_chroma_sub(upper_rgb, lower_rgb, xyz_t, l, h, l_val, h_val, c):
    """
    与えられた条件下での Chroma の限界値を算出する。
    """
    upper_rgb = [
        upper_rgb[idx].subs({l: l_val, h: h_val}) for idx in range(3)]
    lower_rgb = [
        lower_rgb[idx].subs({l: l_val, h: h_val}) for idx in range(3)]
    xyz_t = [
        xyz_t[idx].subs({l: l_val, h: h_val}) for idx in range(3)]

    # まず解く
    upper_solution_zero = [solve(upper_rgb[idx] + 0) for idx in range(3)]
    upper_solution_one = [solve(upper_rgb[idx] - 1) for idx in range(3)]
    lower_solution_zero = [solve(lower_rgb[idx] + 0) for idx in range(3)]
    lower_solution_one = [solve(lower_rgb[idx] - 1) for idx in range(3)]

    # それぞれの解が \sigma の条件を満たしているか確認
    solve_list = []
    for idx in range(3):
        for solve_val in upper_solution_zero[idx]:
            t_val = xyz_t[idx].subs({c: solve_val})
            if t_val > SIGMA:
                solve_list.append(solve_val)
        for solve_val in upper_solution_one[idx]:
            t_val = xyz_t[idx].subs({c: solve_val})
            if t_val > SIGMA:
                solve_list.append(solve_val)

        for solve_val in lower_solution_zero[idx]:
            t_val = xyz_t[idx].subs({c: solve_val})
            if t_val <= SIGMA:
                solve_list.append(solve_val)
        for solve_val in lower_solution_one[idx]:
            t_val = xyz_t[idx].subs({c: solve_val})
            if t_val <= SIGMA:
                solve_list.append(solve_val)

    # 出揃った全てのパターンの中から最小値を選択する
    solve_list = np.array(solve_list)
    chroma = np.min(solve_list[solve_list >= 0.0])

    return chroma


def plot_and_save_ab_plane(idx, data):
    graph_name = "./ab_plane_seq/L_num_{}_{:04d}.png".format(l_sample_num, idx)
    rad = np.linspace(0, 2 * np.pi, h_sample_num)
    a = data * np.cos(rad)
    b = data * np.sin(rad)
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="CIELAB Plane",
        graph_title_size=None,
        xlabel="a*", ylabel="b*",
        axis_label_size=None,
        legend_size=17,
        xlim=(-200, 200),
        ylim=(-200, 200),
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(a, b, label="L*={:.03f}".format(idx * 100 / (l_sample_num - 1)))
    plt.legend(loc='upper left')
    plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    print("plot l_idx={}".format(idx))
    # plt.show()


def visualization_ab_plane():
    """
    ab plane を L* = 0～100 で静止画にして吐く。
    後で Resolve で動画にして楽しもう！
    """
    calc_data = np.load("./data/L_256_H_256_data.npy")
    # for l_idx in range(l_sample_num):
    #     plot_and_save_ab_plane(idx=l_idx, data=calc_data[l_idx])

    args = []

    with Pool(cpu_count()) as pool:
        for l_idx in range(l_sample_num):
            args.append([l_idx, calc_data[l_idx]])
        pool.map(thread_wrapper_visualization, args)


def thread_wrapper_visualization(args):
    return plot_and_save_ab_plane(*args)


def thread_wrapper_visualization_formula(args):
    return plot_formula_for_specific_lstar(*args)


def visualization_formula():
    """
    C* を設定すると RGB 値が求まる数式をプロットする。
    全体的に見つめて問題が無いか確認する。
    """

    # upper_rgb, lower_rgb, xyz_t, l, h, c = lab_to_xyz_formla()
    upper_xyz, lower_xyz, xyz_t, l, h, c = lab_to_xyz_formla_2nd()
    l_vals = np.linspace(0, 100, l_sample_num)
    h_vals = np.linspace(0, 2*np.pi, h_sample_num)
    for l_idx, l_val in enumerate(l_vals):
        args = []
        for h_idx, h_val in enumerate(h_vals):
            args.append([upper_xyz, lower_xyz,
                         l, h, c, l_idx, l_val, h_idx, h_val])
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_visualization_formula, args)


def plot_formula_for_specific_lstar(
        upper_xyz, lower_xyz, l, h, c, l_idx, l_val, h_idx, h_val):
    """
    Q：何をするの？
    A：* l_val, h_val を代入した C* の数式を計算(6本)
       * C* に -250～+250 を代入してプロット＆保存
       * ファイル名は *l_{l_idx}_formula_{h_idx}.png*
    """
    print(l_idx, h_idx)
    # l_val, h_val 代入
    upper_xyz = [
        upper_xyz[idx].subs({l: l_val, h: h_val}) for idx in range(3)]
    lower_xyz = [
        lower_xyz[idx].subs({l: l_val, h: h_val}) for idx in range(3)]
    xyz_t2, c2, l2, h2 = get_xyz_t()
    xyz_t2 = [
        xyz_t2[idx].subs({l2: l_val, h2: h_val}) for idx in range(3)]

    # lambdify 実行
    upper_xyz = [lambdify(c, upper_xyz[idx], 'numpy') for idx in range(3)]
    lower_xyz = [lambdify(c, lower_xyz[idx], 'numpy') for idx in range(3)]
    xyz_t2 = [lambdify(c2, xyz_t2[idx], 'numpy') for idx in range(3)]

    # プロット対象のY軸データ作成
    x = np.linspace(-250, 250, 1024)
    upper_xyz = [upper_xyz[idx](x) for idx in range(3)]
    lower_xyz = [lower_xyz[idx](x) for idx in range(3)]
    xyz_t2 = [xyz_t2[idx](x) for idx in range(3)]

    # XYZ to RGB 変換
    matrix = get_xyz_to_rgb_matrix(primaries=cs.REC2020_xy)
    upper_rgb = apply_matrix(upper_xyz, matrix)
    lower_rgb = apply_matrix(lower_xyz, matrix)

    # 1次元になっちゃうやつへの対処
    for idx in range(3):
        if not isinstance(upper_rgb[idx], np.ndarray):
            upper_rgb[idx] = np.ones_like(x) * upper_rgb[idx]
        if not isinstance(lower_rgb[idx], np.ndarray):
            lower_rgb[idx] = np.ones_like(x) * lower_rgb[idx]
        if not isinstance(xyz_t2[idx], np.ndarray):
            xyz_t2[idx] = np.ones_like(x) * xyz_t2[idx]

    # upper_rgb と lower_rgb を合成
    rgb = [np.ones_like(upper_rgb[idx])*100 for idx in range(3)]
    for idx in range(3):
        upper_idx = xyz_t2[idx] > SIGMA
        lower_idx = xyz_t2[idx] <= SIGMA
        rgb[idx][upper_idx] = upper_rgb[idx][upper_idx]
        rgb[idx][lower_idx] = lower_rgb[idx][lower_idx]

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
    ax1.plot(x, xyz_t2[0] > SIGMA, 'r-', alpha=0.3, label="are")
    ax1.plot(x, xyz_t2[1] > SIGMA, 'g-', alpha=0.3, label="are")
    ax1.plot(x, xyz_t2[2] > SIGMA, 'b-', alpha=0.3, label="are")
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
    ax1.plot(x, xyz_t2[0] > SIGMA, 'r-', alpha=0.3, label="are")
    ax1.plot(x, xyz_t2[1] > SIGMA, 'g-', alpha=0.3, label="are")
    ax1.plot(x, xyz_t2[2] > SIGMA, 'b-', alpha=0.3, label="are")
    plt.legend(loc='upper left')
    plt.savefig(graph_name_1, bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    img_0 = read_image(graph_name_0)
    img_1 = read_image(graph_name_1)
    img = np.hstack((img_0, img_1))
    write_image(img, graph_name_2)
    os.remove(graph_name_0)
    os.remove(graph_name_1)


def lab_to_xyz_formla_2nd():
    """
    数式を取得。XYZ to RGB 変換をしないのが
    初代との違い。
    """
    # base formula
    xyz_t, c, l, h = get_xyz_t()

    # upper
    upper_xyz = [
        get_large_xyz_symbol(n=D65_WHITE[idx], t=xyz_t[idx], upper=True)
        for idx in range(3)]

    # lower
    lower_xyz = [
        get_large_xyz_symbol(n=D65_WHITE[idx], t=xyz_t[idx], upper=False)
        for idx in range(3)]

    # chroma = solve_chroma(upper_rgb, lower_rgb, xyz_t, l, h, l_val, h_val, c)

    return upper_xyz, lower_xyz, xyz_t, l, h, c


def plot_xyz_formula(upper_xyz, lower_xyz, xyz_t,
                     l, h, c, l_idx, l_val, h_idx, h_val):
    print("L*: {}/{}, Hue: {}/{}".format(
        l_idx+1, l_sample_num, h_idx+1, h_sample_num))

    # l_val, h_val 代入
    upper_xyz = [
        upper_xyz[idx].subs({l: l_val, h: h_val}) for idx in range(3)]
    lower_xyz = [
        lower_xyz[idx].subs({l: l_val, h: h_val}) for idx in range(3)]
    xyz_t = [
        xyz_t[idx].subs({l: l_val, h: h_val}) for idx in range(3)]

    # lambdify 実行
    upper_xyz = [lambdify(c, upper_xyz[idx], 'numpy') for idx in range(3)]
    lower_xyz = [lambdify(c, lower_xyz[idx], 'numpy') for idx in range(3)]
    xyz_t = [lambdify(c, xyz_t[idx], 'numpy') for idx in range(3)]

    # プロット対象のY軸データ作成
    x = np.linspace(-250, 250, 1024)
    upper_xyz = [upper_xyz[idx](x) for idx in range(3)]
    lower_xyz = [lower_xyz[idx](x) for idx in range(3)]
    xyz_t = [xyz_t[idx](x) for idx in range(3)]

    # 1次元になっちゃうやつへの対処
    for idx in range(3):
        if not isinstance(upper_xyz[idx], np.ndarray):
            upper_xyz[idx] = np.ones_like(x) * upper_xyz[idx]
        if not isinstance(lower_xyz[idx], np.ndarray):
            lower_xyz[idx] = np.ones_like(x) * lower_xyz[idx]
        if not isinstance(xyz_t[idx], np.ndarray):
            xyz_t[idx] = np.ones_like(x) * xyz_t[idx]

    # upper_rgb と lower_rgb を合成
    large_xyz = [np.ones_like(upper_xyz[idx])*100 for idx in range(3)]
    for idx in range(3):
        upper_idx = xyz_t[idx] > SIGMA
        lower_idx = xyz_t[idx] <= SIGMA
        large_xyz[idx][upper_idx] = upper_xyz[idx][upper_idx]
        large_xyz[idx][lower_idx] = lower_xyz[idx][lower_idx]

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
    ax1.plot(x, large_xyz[0], 'r-', label="R")
    ax1.plot(x, large_xyz[1], 'g-', label="G")
    ax1.plot(x, large_xyz[2], 'b-', label="B")
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
    ax1.plot(x, large_xyz[0], 'r-', label="R")
    ax1.plot(x, large_xyz[1], 'g-', label="G")
    ax1.plot(x, large_xyz[2], 'b-', label="B")
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


def thread_wrapper_visualization_xyz(args):
    plot_xyz_formula(*args)


def visualization_large_xyz_from_lab():
    """
    そもそも、CIELAB to XYZ って滑らかなの？との疑念が出たので
    基本的なところから確認する。
    """
    upper_xyz, lower_xyz, xyz_t, l, h, c = lab_to_xyz_formla_2nd()
    l_vals = np.linspace(0, 100, l_sample_num)
    h_vals = np.linspace(0, 2*np.pi, h_sample_num)

    for l_idx, l_val in enumerate(l_vals):
        args = []
        for h_idx, h_val in enumerate(h_vals):
            args.append([upper_xyz, lower_xyz, xyz_t,
                         l, h, c, l_idx, l_val, h_idx, h_val])
            # plot_xyz_formula(upper_xyz, lower_xyz, xyz_t,
            #                  l, h, c, l_idx, l_val, h_idx, h_val)
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_visualization_xyz, args)


def visualization_rgb_from_lab_using_colour():
    """
    colour を使って Lab to RGB した際の確認。
    """
    l_vals = np.linspace(0, 100, l_sample_num)
    h_vals = np.linspace(0, 2*np.pi, h_sample_num)

    for l_idx, l_val in enumerate(l_vals):
        args = []
        for h_idx, h_val in enumerate(h_vals):
            args.append([l_val, h_val, l_idx, h_idx])
            # plot_lab_to_rgb_using_colour(l_val, h_val, l_idx, h_idx)
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_lab_to_rgb_using_colour, args)


def thread_wrapper_lab_to_rgb_using_colour(args):
    plot_lab_to_rgb_using_colour(*args)


def plot_lab_to_rgb_using_colour(l, h, l_idx, h_idx):
    c = np.linspace(-50, 250, 1024)
    print(np.cos(h))
    a_array = c * np.cos(h)
    b_array = c * np.sin(h)
    l_array = np.ones_like(a_array) * l
    lab = np.dstack((l_array, a_array, b_array))

    # rgb = Lab_to_XYZ(lab)
    large_xyz = Lab_to_XYZ(lab)
    rgb = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65, BT2020_COLOURSPACE.XYZ_to_RGB_matrix)
    # print(rgb)

    # ちょっとデバッグ
    xyz_t2, c2, l2, h2 = get_xyz_t()
    xyz_t2 = [
        xyz_t2[idx].subs({l2: l, h2: h}) for idx in range(3)]
    xyz_t2 = [lambdify(c2, xyz_t2[idx], 'numpy') for idx in range(3)]
    xyz_t2 = [xyz_t2[idx](c) for idx in range(3)]
    for idx in range(3):
        if not isinstance(xyz_t2[idx], np.ndarray):
            xyz_t2[idx] = np.ones_like(c) * xyz_t2[idx]

    graph_name_0 = "./formula_seq/L0_{:03d}_{:04d}.png".format(l_idx, h_idx)
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
    ax1.plot(c, rgb[..., 0].flatten(), 'r-', label="R")
    ax1.plot(c, rgb[..., 1].flatten(), 'g-', label="G")
    ax1.plot(c, rgb[..., 2].flatten(), 'b-', label="B")
    # ax1.plot(c, xyz_t2[0], 'r-', label="X")
    # ax1.plot(c, xyz_t2[1], 'g-', label="Y")
    # ax1.plot(c, xyz_t2[2], 'b-', label="Z")
    # ax1.plot(c, xx2, 'k-', label="My X")
    # ax1.plot(x, xyz_t2[0] > SIGMA, 'r-', alpha=0.3, label="are")
    # ax1.plot(x, xyz_t2[1] > SIGMA, 'g-', alpha=0.3, label="are")
    # ax1.plot(x, xyz_t2[2] > SIGMA, 'b-', alpha=0.3, label="are")
    plt.legend(loc='upper left')
    plt.savefig(graph_name_0, bbox_inches='tight', pad_inches=0.1)
    # plt.show()


def experimental_functions():
    # check_basic_trigonometricfunction()
    # plot_inv_f()
    # plot_ab_plane()
    # start = time.time()
    # solve_chroma()
    # end = time.time()
    # print("time = {:.4f} [s]".format(end - start))
    # data = np.array(shared_array[:]).reshape((l_sample_num, h_sample_num))
    # fname = "./data/L_{}_H_{}_data.npy".format(l_sample_num, h_sample_num)
    # np.save(fname, data)
    # visualization_ab_plane()
    # visualization_formula()
    # visualization_large_xyz_from_lab()
    visualization_rgb_from_lab_using_colour()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    experimental_functions()
    # matrix = get_xyz_to_rgb_matrix(primaries=cs.REC709_xy)
    # print(matrix)
    """
    time = 7.8243 [s]
    [[0.0 0.0 0.0]
    [105.609084208454 77.1514136635651 105.609084208454]
    [0.0 1.02149424848761e-13 2.28617836127562e-11]]
    """
