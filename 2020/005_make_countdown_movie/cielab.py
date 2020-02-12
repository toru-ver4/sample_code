# -*- coding: utf-8 -*-
"""
CIELAB の特殊操作用
=====================

"""

# import standard libraries
import os
import time
import ctypes

# import third-party libraries
# import matplotlib as mpl
# mpl.use('Agg')
import numpy as np
from sympy import symbols, sin, cos, lambdify
from sympy.solvers import solve
from scipy import linalg
from colour.models import BT2020_COLOURSPACE, BT709_COLOURSPACE
from colour import xy_to_XYZ, read_image, write_image, Lab_to_XYZ, XYZ_to_RGB,\
    LUT3D, RGB_to_XYZ, XYZ_to_Lab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool, cpu_count, Array
import cv2

# import my libraries
import color_space as cs
import plot_utility as pu

# definition
D65_WHITE = xy_to_XYZ(cs.D65) * 100
D65_X = D65_WHITE[0]
D65_Y = D65_WHITE[1]
D65_Z = D65_WHITE[2]

# D65_X = 95.04
# D65_Y = 100.0
# D65_Z = 108.89
# D65_WHITE = [D65_X, D65_Y, D65_Z]
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
l_sample_num = 256
h_sample_num = 256
shared_array = Array(
    typecode_or_type=ctypes.c_float,
    size_or_initializer=l_sample_num*h_sample_num)
npy_name = "chroma_l_{}_h_{}.npy".format(l_sample_num, h_sample_num)
im_threshold = 0.0000001


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


def lab_to_rgb_expr(l, c, h, primaries=cs.REC2020_xy):
    mtx = get_xyz_to_rgb_matrix(primaries=primaries)
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
    l_idx = 1
    h_idx = 12
    plot_formula_for_specific_lstar(
        l_vals[l_idx], l_idx, h_vals[h_idx], h_idx, rgb_exprs, l, c, h)
    return None
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
    print("l_idx={}, h_idx={}".format(l_idx, h_idx))
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
    arg_idxs = [np.zeros_like(rgb_exprs[0][idx]) for idx in range(3)]
    for ii in range(len(IJK_LIST)):
        for jj in range(3):
            x_idx = (xyz_t[0] > SIGMA) if IJK_LIST[ii][0] else (xyz_t[0] <= SIGMA)
            y_idx = (xyz_t[1] > SIGMA) if IJK_LIST[ii][1] else (xyz_t[1] <= SIGMA)
            z_idx = (xyz_t[2] > SIGMA) if IJK_LIST[ii][2] else (xyz_t[2] <= SIGMA)
            idx = (x_idx & y_idx) & z_idx
            rgb[jj][idx] = rgb_exprs[ii][jj][idx]
            arg_idxs[jj][idx] = ii * 0.1

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
        ylim=(-0.5, 0.8),
        xtick=[25 * x - 50 for x in range(13)],
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(x, rgb[0], 'r-', label="R")
    ax1.plot(x, rgb[1], 'g-', label="G")
    ax1.plot(x, rgb[2], 'b-', label="B")
    # ax1.plot(x, arg_idxs[0], 'r-', alpha=0.3, label="arg_idx")
    # ax1.plot(x, arg_idxs[1], 'g-', alpha=0.3, label="arg_idx")
    ax1.plot(x, arg_idxs[2], 'b-', alpha=0.3, label="arg_idx")
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


def solve_chroma(l_val, h_val, rgb_exprs, l, c, h):
    """
    引数で与えられた L*, H に対する Chroma値を算出する。
    ```make_chroma_array``` のループからコールされることが前提のコード。
    """
    # start = time.time()
    xyz_t = [get_tx(l, c, h), get_ty(l), get_tz(l, c, h)]
    xyz_t = [xyz_t[idx].subs({l: l_val, h: h_val}) for idx in range(3)]
    temp_solution = []

    for ii in range(len(IJK_LIST)):
        for jj in range(3):  # R, G, B のループ
            # l_val, h_val 代入
            c_expr = rgb_exprs[ii][jj].subs({l: l_val, h: h_val})
            solution = []
            solution.extend(solve(c_expr))
            solution.extend(solve(c_expr - 1))

            for solve_val_complex in solution:
                # 複素成分を見て、小さければ実数とみなす
                # どうも solve では複素数として算出されてしまうケースがあるっぽい
                solve_val, im_val = solve_val_complex.as_real_imag()
                if abs(im_val) > im_threshold:
                    continue

                t = [xyz_t[kk].subs({c: solve_val}) for kk in range(3)]
                xt_bool = (t[0] > SIGMA) if IJK_LIST[ii][0] else (t[0] <= SIGMA)
                yt_bool = (t[1] > SIGMA) if IJK_LIST[ii][1] else (t[1] <= SIGMA)
                zt_bool = (t[2] > SIGMA) if IJK_LIST[ii][2] else (t[2] <= SIGMA)
                xyz_t_bool = (xt_bool and yt_bool) and zt_bool
                if xyz_t_bool:
                    temp_solution.append(solve_val)

    chroma_list = np.array(temp_solution)
    chroma = np.min(chroma_list[chroma_list >= 0.0])

    print("L*={:.2f}, H={:.2f}, C={:.3f}".format(
            l_val, h_val / (2 * np.pi) * 360, chroma))
    # end = time.time()
    # print("each_time={}[s]".format(end-start))
    return chroma


def solve_chroma_wrapper(args):
    solve_chroma(*args)


def make_chroma_array():
    """
    L*a*b* 空間における a*b*平面の境界線プロットのために、
    各L* における 境界線の Chroma を計算する。
    """
    l, c, h = symbols('l, c, h')
    rgb_exprs = lab_to_rgb_expr(l, c, h)
    l_vals = np.linspace(0, 100, l_sample_num)
    h_vals = np.linspace(0, 2*np.pi, h_sample_num)
    for l_idx, l_val in enumerate(l_vals):
        args = []
        for h_idx, h_val in enumerate(h_vals):
            args.append([l_val, l_idx, h_val, h_idx, rgb_exprs, l, c, h])
        with Pool(cpu_count()) as pool:
            pool.map(solve_chroma_wrapper, args)

    chroma = np.array(shared_array[:]).reshape((l_sample_num, h_sample_num))
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


def plot_and_save_ab_plane_color(idx, data):
    graph_name = "./ab_plane_seq/color_L_num_{}_{:04d}.png".format(
        l_sample_num, idx)
    rad = np.linspace(0, 2 * np.pi, h_sample_num)
    a = data * np.cos(rad)
    b = data * np.sin(rad)
    large_l = np.ones_like(a) * (idx * 100) / (l_sample_num - 1)
    lab = np.dstack((large_l, a, b)).reshape((h_sample_num, 3))
    large_xyz = Lab_to_XYZ(lab)
    rgb = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65, BT2020_COLOURSPACE.XYZ_to_RGB_matrix)
    rgb = np.clip(rgb, 0.0, 1.0) ** (1/2.4)

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="CIELAB Plane L*={:.03f}".format(
            idx * 100 / (l_sample_num - 1)),
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
    # ax1.plot(a, b, label="L*={:.03f}".format(idx * 100 / (l_sample_num - 1)))
    ax1.patch.set_facecolor("#E0E0E0")
    ax1.scatter(a, b, c=rgb)
    # plt.legend(loc='upper left')
    plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    print("plot l_idx={}".format(idx))
    # plt.show()


def plot_and_save_ab_plane_fill_color(idx, data, inner_rgb, inner_lab):
    graph_name = "./ab_plane_seq/verify_L_num_{}_{:04d}.png".format(
        l_sample_num, idx)
    rad = np.linspace(0, 2 * np.pi, h_sample_num)
    a = data * np.cos(rad)
    b = data * np.sin(rad)
    large_l = np.ones_like(a) * (idx * 100) / (l_sample_num - 1)
    lab = np.dstack((large_l, a, b)).reshape((h_sample_num, 3))
    large_xyz = Lab_to_XYZ(lab)
    rgb = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65, BT2020_COLOURSPACE.XYZ_to_RGB_matrix)
    rgb = np.clip(rgb, 0.0, 1.0) ** (1/2.4)

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="CIELAB Plane L*={:.03f}".format(
            idx * 100 / (l_sample_num - 1)),
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
    # ax1.plot(a, b, label="L*={:.03f}".format(idx * 100 / (l_sample_num - 1)))
    ax1.patch.set_facecolor("#B0B0B0")
    # ax1.scatter(a, b, c=rgb)
    ax1.plot(a, b, '-k')
    ax1.scatter(inner_lab[..., 1], inner_lab[..., 2], c=inner_rgb, s=7.5)
    # plt.legend(loc='upper left')
    plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    print("plot l_idx={}".format(idx))
    # plt.show()


def visualization_ab_plane_fill_color(sample=256):
    """
    ab plane を L* = 0～100 で静止画にして吐く。
    後で Resolve で動画にして楽しもう！
    """
    calc_data = np.load(npy_name)
    delta_l = 0.001 * 100

    rgb = LUT3D.linear_table(sample).reshape((1, sample ** 3, 3)) ** (2.4)
    xyz = RGB_to_XYZ(rgb, cs.D65, cs.D65, BT2020_COLOURSPACE.RGB_to_XYZ_matrix)
    lab = XYZ_to_Lab(xyz)
    rgb = rgb ** (1/2.4)

    args = []
    l_list = np.linspace(0, 100, l_sample_num)

    with Pool(cpu_count()) as pool:
        for l_idx, l_val in enumerate(l_list):
            ok_idx = (l_val - delta_l <= lab[:, :, 0]) & (lab[:, :, 0] < l_val + delta_l)
            args.append([l_idx, calc_data[l_idx], rgb[ok_idx], lab[ok_idx]])
        pool.map(thread_wrapper_visualization_ab_plane_fill_color, args)


def thread_wrapper_visualization_ab_plane_fill_color(args):
    return plot_and_save_ab_plane_fill_color(*args)


def visualization_ab_plane_color(sample=256):
    """
    ab plane を L* = 0～100 で静止画にして吐く。
    後で Resolve で動画にして楽しもう！
    """
    calc_data = np.load(npy_name)

    args = []
    l_list = np.linspace(0, 100, l_sample_num)

    with Pool(cpu_count()) as pool:
        for l_idx, l_val in enumerate(l_list):
            args.append([l_idx, calc_data[l_idx]])
        pool.map(thread_wrapper_visualization_ab_plane_color, args)


def thread_wrapper_visualization_ab_plane_color(args):
    return plot_and_save_ab_plane_color(*args)


def visualization_cielab_color_volume():
    chroma = np.load(npy_name)
    args = []

    with Pool(cpu_count()) as pool:
        for l_idx in range(l_sample_num):
            args.append([l_idx, chroma])
        pool.map(thread_wrapper_cielab_color_volume, args)


def thread_wrapper_cielab_color_volume(args):
    plot_cielab_color_volume_seq(*args)


def plot_cielab_color_volume_seq(l_idx, chroma):
    graph_name = "./Lab_color_volume_seq/no_{:04d}.png".format(l_idx)
    print("l_idx = {}".format(l_idx))

    rad = np.linspace(0, 2 * np.pi, h_sample_num)
    ll = np.linspace(0, 100, l_sample_num).reshape((l_sample_num, 1))
    ll = ll[:l_idx+1]
    chroma[-1, :] = 0.0
    a = chroma[:l_idx+1] * np.cos(rad) + cs.D65[0]
    b = chroma[:l_idx+1] * np.sin(rad) + cs.D65[1]
    ly = np.ones_like(b) * ll
    large_xyz = Lab_to_XYZ(np.dstack((ly, a, b)))
    rgb = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65, BT2020_COLOURSPACE.XYZ_to_RGB_matrix)
    rgb = np.clip(rgb, 0.0, 1.0) ** (1/2.4)
    rgb = rgb.reshape(((l_idx + 1) * h_sample_num, 3))

    fig = plt.figure(figsize=(9, 9))
    ax = Axes3D(fig)
    ax.set_xlabel("a*")
    ax.set_ylabel("b*")
    ax.set_zlabel("L*")
    ax.set_title("L*={:.03f} CIELAB Color Volume".format(ll[-1, 0]),
                 fontsize=18)
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.set_zlim(0, 110)
    ax.view_init(elev=20, azim=-120)
    ax.scatter(a.flatten(), b.flatten(), ly.flatten(),
               marker='o', c=rgb, zorder=1)
    plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    # plt.show()


def resize_and_hstack(fname1, fname2, fname3):
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


def visualization_cielab_color_volume_seq():
    """
    a*b* plane と L*a*b* のプロットを連番静止画にする。
    """
    args = []
    with Pool(cpu_count()) as pool:
        for l_idx in range(l_sample_num):
            volume_name = "./Lab_color_volume_seq/no_{:04d}.png".format(l_idx)
            plane_name = "./ab_plane_seq/color_L_num_{}_{:04d}.png".format(
                l_sample_num, l_idx)
            concat_name = "./Lab_color_volume_seq/concat_{:04d}.png".format(
                l_idx)
            args.append([volume_name, plane_name, concat_name])
        pool.map(thread_wrapper_visualization_cielab_color_volume_seq, args)


def thread_wrapper_visualization_cielab_color_volume_seq(args):
    resize_and_hstack(*args)


def experimental_functions():
    # visualize_formula()  # 事前調査用
    # chroma = make_chroma_array()
    # np.save(npy_name, chroma)
    # visualization_ab_plane_color()
    # visualization_ab_plane_fill_color()
    # visualization_cielab_color_volume()
    # visualization_cielab_color_volume_seq()
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    start = time.time()
    experimental_functions()
    end = time.time()
    print("total_time={}[s]".format(end-start))
