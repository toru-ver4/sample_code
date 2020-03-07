# -*- coding: utf-8 -*-
"""
Title
==============

Description.

"""

# import standard libraries
import os
import ctypes

# import third-party libraries
from sympy import symbols
import numpy as np
from multiprocessing import Pool, cpu_count, Array
import matplotlib.pyplot as plt


# import my libraries
import cielab as cl
import color_space as cs
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

L_SAMPLE_NUM_MAX = 1024
H_SAMPLE_NUM_MAX = 1024

shared_array = Array(
    typecode_or_type=ctypes.c_float,
    size_or_initializer=L_SAMPLE_NUM_MAX*H_SAMPLE_NUM_MAX)


def plot_and_save_ab_plane(idx, data, l_sample_num, h_sample_num):
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
    print("plot l_idx={}".format(idx))
    plt.show()


def solve_chroma_wrapper(args):
    chroma = cl.solve_chroma(**args)
    s_idx = args['h_sample_num'] * args['l_idx'] + args['h_idx']
    shared_array[s_idx] = chroma


def make_chroma_array(primaries=cs.get_primaries(cs.BT709),
                      l_sample_num=L_SAMPLE_NUM_MAX,
                      h_sample_num=H_SAMPLE_NUM_MAX):
    """
    L*a*b* 空間における a*b*平面の境界線プロットのために、
    各L* における 境界線の Chroma を計算する。
    """

    l, c, h = symbols('l, c, h')
    rgb_exprs = cl.lab_to_rgb_expr(l, c, h, primaries=primaries)
    l_vals = np.linspace(0, 100, l_sample_num)
    h_vals = np.linspace(0, 2*np.pi, h_sample_num)
    for l_idx, l_val in enumerate(l_vals):
        args = []
        for h_idx, h_val in enumerate(h_vals):
            d = dict(
                l_val=l_val, l_idx=l_idx, h_val=h_val, h_idx=h_idx,
                rgb_exprs=rgb_exprs, l=l, c=c, h=h,
                l_sample_num=l_sample_num, h_sample_num=h_sample_num)
            args.append(d)
        with Pool(cpu_count()) as pool:
            pool.map(solve_chroma_wrapper, args)

    chroma = np.array(
        shared_array[:l_sample_num * h_sample_num]).reshape(
            (l_sample_num, h_sample_num))
    return chroma


def make_bt709_cielab_outline_data(
        l_sample_num=L_SAMPLE_NUM_MAX, h_sample_num=H_SAMPLE_NUM_MAX):
    chroma = make_chroma_array(
        primaries=cs.get_primaries(cs.BT709),
        l_sample_num=l_sample_num, h_sample_num=h_sample_num)
    fname = f"Chroma_BT709_l_{l_sample_num}_h_{h_sample_num}.npy"
    np.save(fname, chroma)


def make_bt2020_cielab_outline_data(
        l_sample_num=L_SAMPLE_NUM_MAX, h_sample_num=H_SAMPLE_NUM_MAX):
    chroma = make_chroma_array(
        primaries=cs.get_primaries(cs.BT2020),
        l_sample_num=l_sample_num, h_sample_num=h_sample_num)
    fname = f"Chroma_BT2020_l_{l_sample_num}_h_{h_sample_num}.npy"
    np.save(fname, chroma)


def make_ab_plane_boundary_data(
        lstar=50, h_sample_num=256, color_space=cs.BT709):
    l, c, h = symbols('l, c, h')
    primaries = cs.get_primaries(color_space)
    rgb_exprs = cl.lab_to_rgb_expr(l, c, h, primaries=primaries)
    l_val = lstar
    h_vals = np.linspace(0, 2*np.pi, h_sample_num)
    args = []
    for h_idx, h_val in enumerate(h_vals):
        d = dict(
            l_val=l_val, l_idx=0, h_val=h_val, h_idx=h_idx,
            rgb_exprs=rgb_exprs, l=l, c=c, h=h,
            l_sample_num=0, h_sample_num=h_sample_num)
        args.append(d)
    with Pool(cpu_count()) as pool:
        pool.map(solve_chroma_wrapper, args)

    chroma = np.array(shared_array[:h_sample_num])
    fname = f"Chroma_L_{lstar}_BT709_h_{h_sample_num}.npy"
    np.save(fname, chroma)
    return chroma


def main_func():
    # L*a*b* 全体のデータを算出
    l_sample_num = 5
    h_sample_num = 32
    make_bt709_cielab_outline_data(
        l_sample_num=l_sample_num, h_sample_num=h_sample_num)
    make_bt2020_cielab_outline_data(
        l_sample_num=l_sample_num, h_sample_num=h_sample_num)

    # 任意の L* の a*b*平面用の boundary データ計算


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # main_func()
    # make_ab_plane_boundary_data()

    # primaries = cs.get_primaries(cs.BT709)
    # print(primaries)
    # l_sample_num = 5
    # h_sample_num = 32
    # fname = f"Chroma_BT709_l_{l_sample_num}_h_{h_sample_num}.npy"
    # chroma = np.load(fname)
    # plot_and_save_ab_plane(1, chroma[1], l_sample_num, h_sample_num)
    # fname = f"Chroma_BT2020_l_{l_sample_num}_h_{h_sample_num}.npy"
    # chroma = np.load(fname)
    # plot_and_save_ab_plane(1, chroma[1], l_sample_num, h_sample_num)

    lstar = 50
    h_sample_num = 256
    fname = f"Chroma_L_{lstar}_BT709_h_{h_sample_num}.npy"
    chroma = np.load(fname)
    plot_and_save_ab_plane(1, chroma, 0, h_sample_num)
