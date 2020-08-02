# -*- coding: utf-8 -*-
"""
investigate the crosstalk matrix
================================

"""

# import standard libraries
import os
import time
from multiprocessing import Pool, cpu_count

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from colour import XYZ_to_RGB, xyY_to_XYZ, RGB_COLOURSPACES,\
    RGB_to_XYZ, XYZ_to_xyY
from colour.models import BT2020_COLOURSPACE

# import my libraries
import color_space as cs
import bt2446_method_c as bmc
import test_pattern_generator2 as tpg
import plot_utility as pu
from bt2446_method_c import bt2446_method_c_tonemapping

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


Y_NUM = 512
H_NUM = 2048


class MeasureExecTime():
    def __init__(self):
        self.clear_buf()

    def clear_buf(self):
        self.st_time = 0.0
        self.lap_st = 0.0
        self.ed_time = 00

    def start(self):
        self.st_time = time.time()
        self.lap_st = self.st_time

    def lap(self):
        current = time.time()
        print(f"lap time = {current - self.lap_st:.5f} [sec]")
        self.lap_st = current

    def end(self):
        current = time.time()
        print(f"total time = {current - self.st_time:.5f} [sec]")
        self.clear_buf()


def plot_xy_plane_specific_y(
        y_idx, y_num, h_num, lut, prefix_str=""):
    graph_name = "/work/overuse/2020/022_bt2446/xy_plane/"\
        + f"{prefix_str}_no_{y_idx:04d}.png"
    large_y = y_idx / (y_num - 1)
    xx = lut[y_idx][..., 0]
    yy = lut[y_idx][..., 1]
    llyy = np.ones_like(xx) * large_y

    large_xyz = xyY_to_XYZ(np.dstack((xx, yy, llyy)).reshape(h_num, 3))
    rgb = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65, BT2020_COLOURSPACE.XYZ_to_RGB_matrix)
    rgb = np.clip(rgb, 0.0, 1.0) ** (1/2.4)
    cmf_xy = tpg._get_cmfs_xy()

    fig, ax1 = pu.plot_1_graph(
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
        return_figure=True)
    ax1.patch.set_facecolor("#B0B0B0")
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=3, label=None)
    ax1.plot((cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
             '-k', lw=3, label=None)

    # ax1.plot(x, y, 'k-')
    ax1.scatter(xx, yy, c=rgb)
    # plt.legend(loc='upper left')
    plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    print("plot y_idx={}".format(y_idx))
    plt.close(fig)


def plot_xy_plane_specific_y_with_bt2446(
        y_idx, y_num, h_num, lut, xyY, xyY_w_ctm, prefix_str="tm"):
    graph_name = "/work/overuse/2020/022_bt2446/xy_plane/"\
        + f"{prefix_str}_no_{y_idx:04d}.png"
    large_y = y_idx / (y_num - 1)
    xx = lut[y_idx][..., 0]
    yy = lut[y_idx][..., 1]
    llyy = np.ones_like(xx) * large_y

    large_xyz = xyY_to_XYZ(np.dstack((xx, yy, llyy)).reshape(h_num, 3))
    rgb = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65, BT2020_COLOURSPACE.XYZ_to_RGB_matrix)
    rgb = np.clip(rgb, 0.0, 1.0) ** (1/2.4)
    cmf_xy = tpg._get_cmfs_xy()

    fig, ax1 = pu.plot_1_graph(
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
        return_figure=True)
    ax1.patch.set_facecolor("#B0B0B0")
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=3, label=None)
    ax1.plot((cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
             '-k', lw=3, label=None)

    ax1.scatter(xx, yy, c=rgb)

    y_idx2 = np.sum(xyY[..., 0, 2] < large_y)
    x2 = xyY[y_idx2][..., 0]
    y2 = xyY[y_idx2][..., 1]
    ax1.plot(x2, y2, '--k', lw=3, label="TM w/o crosstalk matrix")

    y_idx3 = np.sum(xyY_w_ctm[..., 0, 2] < large_y)
    x3 = xyY[y_idx3][..., 0]
    y3 = xyY[y_idx3][..., 1]
    ax1.plot(x3, y3, '-k', lw=3, label="TM w/ crosstalk matrix")

    plt.legend(loc='upper left')
    plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    print("plot y_idx={}".format(y_idx))
    plt.close(fig)


def make_xy_plane_color_image(
        large_y, samples=1024, bg_color=0.5):
    x_list = np.linspace(0.0, 0.8, samples)
    y_list = np.linspace(0.0, 0.9, samples)
    xx, yy = np.meshgrid(x_list, y_list)

    if large_y == 0.0:
        llyy = np.ones_like(xx) * 0.00000000001
    else:
        llyy = np.ones_like(xx) * large_y

    xyY = np.dstack((xx, yy, llyy))
    rgb = XYZ_to_RGB(
        xyY_to_XYZ(xyY), cs.D65, cs.D65,
        BT2020_COLOURSPACE.XYZ_to_RGB_matrix)

    r_ok = (rgb[..., 0] >= 0) & (rgb[..., 0] <= 1.0)
    g_ok = (rgb[..., 1] >= 0) & (rgb[..., 1] <= 1.0)
    b_ok = (rgb[..., 2] >= 0) & (rgb[..., 2] <= 1.0)
    rgb_ok = (r_ok & g_ok) & b_ok

    rgb[~rgb_ok] = np.ones_like(rgb[~rgb_ok]) * bg_color
    rgb = rgb[::-1]

    rgb = rgb ** (1/2.4)

    return rgb


def plot_xy_plane_specific_y_with_bg_color(y_idx, y_num, h_num, lut):
    graph_name = "/work/overuse/2020/022_bt2446/xy_plane/"\
        + f"with_bg_color_no_{y_idx:04d}.png"
    large_y = y_idx / (y_num - 1)
    xx = lut[y_idx][..., 0]
    yy = lut[y_idx][..., 1]

    rgb_img = make_xy_plane_color_image(
        large_y, samples=1024, bg_color=0.5)
    cmf_xy = tpg._get_cmfs_xy()

    fig, ax1 = pu.plot_1_graph(
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
        return_figure=True)
    ax1.patch.set_facecolor("#B0B0B0")
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=3, label=None)
    ax1.plot((cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
             '-k', lw=3, label=None)

    # ax1.plot(x, y, 'k-')
    ax1.scatter(xx, yy, c='k', s=4)
    ax1.imshow(
        rgb_img, extent=(0.0, 0.8, 0.0, 0.9), aspect='auto')
    # plt.legend(loc='upper left')
    plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    print("plot y_idx={}".format(y_idx))
    plt.close(fig)


def plot_xy_plane_specific_y_bt2446_with_bg_color(y_idx, y_val):
    graph_name = "/work/overuse/2020/022_bt2446/xy_plane/"\
        + f"xy_bt2446_with_bg_color_no_{y_idx:04d}.png"
    y_sdr_nits = y_val * 100
    y_hdr_nits = bmc.bt2446_method_c_inverse_tonemapping_core(
        x=y_sdr_nits, k1=0.51, k3=0.75, y_sdr_ip=51.1)
    xx_yy = make_xyY_boundary_data_specific_Y(
        large_y=y_hdr_nits/10000, h_num=H_NUM)
    xx = xx_yy[..., 0]
    yy = xx_yy[..., 1]

    rgb_img = make_xy_plane_color_image(
        y_val, samples=1024, bg_color=0.5)
    cmf_xy = tpg._get_cmfs_xy()

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(8, 9),
        graph_title=f"{y_hdr_nits:.2f} cd/m2 to {y_sdr_nits:.2f} cd/m2",
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
        return_figure=True)
    ax1.patch.set_facecolor("#B0B0B0")
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=3, label=None)
    ax1.plot((cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
             '-k', lw=3, label=None)

    # ax1.plot(x, y, 'k-')
    ax1.plot(xx, yy, '--', c='k', lw=2, label="Gamut boundary after TM")
    ax1.imshow(
        rgb_img, extent=(0.0, 0.8, 0.0, 0.9), aspect='auto')
    plt.legend(loc='upper left')
    plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    print("plot y_idx={}".format(y_idx))
    plt.close(fig)


def is_inner_gamut(xyY, color_space_name=cs.BT2020, white=cs.D65):
    rgb = XYZ_to_RGB(
        xyY_to_XYZ(xyY), white, white,
        RGB_COLOURSPACES[color_space_name].XYZ_to_RGB_matrix)
    r_judge = (rgb[..., 0] >= 0) & (rgb[..., 0] <= 1)
    g_judge = (rgb[..., 1] >= 0) & (rgb[..., 1] <= 1)
    b_judge = (rgb[..., 2] >= 0) & (rgb[..., 2] <= 1)
    judge = (r_judge & g_judge) & b_judge

    return judge


def make_xyY_boundary_data_specific_Y(
        large_y=0.5, color_space_name=cs.BT2020,
        white=cs.D65, h_num=1024):

    if large_y <= 0.0:
        return np.zeros((h_num, 2))

    r_val_init = 0.8
    iteration_num = 20
    hue = np.linspace(0, 2*np.pi, h_num)
    rr = np.ones(h_num) * r_val_init
    llyy = np.ones(h_num) * large_y

    for idx in range(iteration_num):
        xx = rr * np.cos(hue) + white[0]
        yy = rr * np.sin(hue) + white[1]
        xyY = np.dstack((xx, yy, llyy)).reshape((h_num, 3))
        ok_idx = is_inner_gamut(
            xyY=xyY, color_space_name=color_space_name, white=white)

        add_sub = r_val_init / (2 ** (idx + 1))
        rr[ok_idx] = rr[ok_idx] + add_sub
        rr[~ok_idx] = rr[~ok_idx] - add_sub

    xx = rr * np.cos(hue) + white[0]
    yy = rr * np.sin(hue) + white[1]

    return np.dstack((xx, yy)).reshape((h_num, 2))


def make_xyY_boundary_data(
        color_space_name=cs.BT2020, white=cs.D65, y_num=1024, h_num=1024):
    """
    Returns
    -------
    ndarray
        small x and small y for each large Y and hue.
        the shape is (N, M, 2).
        N is a number of large Y.
        M is a number of Hue.
        "2" are small x and small y.
    """
    mtime = MeasureExecTime()
    mtime.start()
    out_buf = np.zeros((y_num, h_num, 2))
    y_list = np.linspace(0, 1.0, y_num)
    for idx, large_y in enumerate(y_list):
        print(f"idx = {idx} / {y_num}")
        out_buf[idx] = make_xyY_boundary_data_specific_Y(
            large_y=large_y, color_space_name=color_space_name,
            white=white, h_num=h_num)
        mtime.lap()
    mtime.end()

    fname = f"./_debug_lut/xyY_LUT_YxH_{y_num}x{h_num}.npy"
    np.save(fname, out_buf)


def make_xyY_boundary_data2(
        color_space_name=cs.BT2020, white=cs.D65, y_num=1024, h_num=1024):
    """
    Returns
    -------
    ndarray
        small x and small y for each large Y and hue.
        the shape is (N, M, 2).
        N is a number of large Y.
        M is a number of Hue.
        "2" are small x and small y.
    """
    mtime = MeasureExecTime()
    mtime.start()
    y_max = bmc.bt2446_method_c_tonemapping_core(
        10000, k1=0.51, k3=0.75, y_sdr_ip=51.1)
    out_buf = np.zeros((y_num, h_num, 2))
    y_list_sdr = np.linspace(0, y_max, y_num)
    y_list_hdr = bmc.bt2446_method_c_inverse_tonemapping_core(
        x=y_list_sdr, k1=0.51, k3=0.75, y_sdr_ip=51.1)
    y_list = y_list_hdr / 10000
    for idx, large_y in enumerate(y_list):
        print(f"make lut, largeY = {large_y:.2} nits, idx = {idx} / {y_num}")
        out_buf[idx] = make_xyY_boundary_data_specific_Y(
            large_y=large_y, color_space_name=color_space_name,
            white=white, h_num=h_num)
        mtime.lap()
    mtime.end()

    return out_buf


def plot_xy_plane_specific_y_wrapper(args):
    plot_xy_plane_specific_y(**args)


def plot_xy_plane_specific_y_with_bg_color_wrapper(args):
    plot_xy_plane_specific_y_with_bg_color(**args)


def plot_xyY_color_volume_seq_wrapper(args):
    plot_xyY_color_volume_seq(**args)


def plot_xy_plane(y_num=2048, h_num=2048):
    lut_name = f"./_debug_lut/xyY_LUT_YxH_{y_num}x{h_num}.npy"
    lut = np.load(lut_name)
    args = []
    # for y_idx in range(y_num):
        # plot_xy_plane_specific_y(
        #     y_idx=y_idx, y_num=y_num, h_num=h_num, lut=lut)
        # plot_xy_plane_specific_y_with_bg_color(
        #     y_idx=y_idx, y_num=y_num, h_num=h_num, lut=lut)
        # d = dict(y_idx=y_idx, y_num=y_num, h_num=h_num, lut=lut)
        # args.append(d)
    # with Pool(cpu_count()) as pool:
    #     pool.map(plot_xy_plane_specific_y_wrapper, args)
    # with Pool(cpu_count()) as pool:
    #     pool.map(plot_xy_plane_specific_y_with_bg_color_wrapper, args)

    temp = tpg.equal_devision(512, 10)
    idx_list = [0] + [np.sum(temp[:x+1]) for x in range(len(temp))]

    for idx in range(len(idx_list) - 1):
        args = []
        for y_idx in range(idx_list[idx], idx_list[idx+1]):
            d = dict(y_idx=y_idx, lut=lut)
            args.append(d)
            # plot_xyY_color_volume_seq(**d)
        with Pool(cpu_count()) as pool:
            pool.map(plot_xyY_color_volume_seq_wrapper, args)


def conv_yhxy_to_rgb(lut, y_num, h_num):
    xx = lut[..., 0]
    yy = lut[..., 1]
    buf = []
    y_values = np.linspace(0, 1, y_num)
    for y_val in y_values:
        buf.append(np.ones(h_num) * y_val)
    llyy = np.vstack(buf)

    xyY = np.dstack((xx, yy, llyy)).reshape((y_num, h_num, 3))
    rgb = XYZ_to_RGB(
        xyY_to_XYZ(xyY), cs.D65, cs.D65,
        BT2020_COLOURSPACE.XYZ_to_RGB_matrix)

    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def plot_xy_plane_specific_y_with_bt2446_wrapper(args):
    plot_xy_plane_specific_y_with_bt2446(**args)


# def plot_xy_plane_with_bt2446(y_num=Y_NUM, h_num=H_NUM):
#     lut_name = f"./_debug_lut/xyY_LUT_YxH_{y_num}x{h_num}.npy"
#     lut = np.load(lut_name)
#     rgb = conv_yhxy_to_rgb(lut, y_num, h_num)

#     rgb_wo_ctm = bt2446_method_c_tonemapping(
#         img=rgb, alpha=0.0, sigma=0.0,
#         hdr_ref_luminance=203, hdr_peak_luminance=1000,
#         k1=0.51, k3=0.75, y_sdr_ip=51.1)
#     xyY_wo_ctm = XYZ_to_xyY(
#         RGB_to_XYZ(rgb_wo_ctm, cs.D65, cs.D65,
#                    BT2020_COLOURSPACE.RGB_to_XYZ_matrix))

#     rgb_w_ctm = bt2446_method_c_tonemapping(
#         img=rgb, alpha=0.2, sigma=0.0,
#         hdr_ref_luminance=203, hdr_peak_luminance=1000,
#         k1=0.51, k3=0.75, y_sdr_ip=51.1)
#     xyY_w_ctm = XYZ_to_xyY(
#         RGB_to_XYZ(rgb_w_ctm, cs.D65, cs.D65,
#                    BT2020_COLOURSPACE.RGB_to_XYZ_matrix))
#     args = []
#     for y_idx in range(y_num):
#         # plot_xy_plane_specific_y_with_bt2446(
#         #     y_idx, y_num, h_num, lut, xyY=xyY_wo_ctm, prefix_str="tm")
#         d = dict(
#             y_idx=y_idx, y_num=y_num, h_num=h_num, lut=lut,
#             xyY=xyY_wo_ctm, xyY_w_ctm=xyY_w_ctm, prefix_str="tm")
#         args.append(d)
#     with Pool(cpu_count()) as pool:
#         pool.map(plot_xy_plane_specific_y_with_bt2446_wrapper, args)


def plot_xy_plane_specific_y_bt2446_with_bg_color_wrapper(args):
    plot_xy_plane_specific_y_bt2446_with_bg_color(**args)


def plot_xyY_color_volume_seq_tm_wrapper(args):
    plot_xyY_color_volume_seq_tm(**args)


def plot_xy_plane_with_bt2446(y_num=Y_NUM, h_num=H_NUM):
    y_max = bmc.bt2446_method_c_tonemapping_core(
        10000, k1=0.51, k3=0.75, y_sdr_ip=51.1) / 100
    y_val_list = np.linspace(0, y_max, y_num)
    args = []
    for y_idx, y_val in enumerate(y_val_list):
        d = dict(y_idx=y_idx, y_val=y_val)
        args.append(d)
        # plot_xy_plane_specific_y_bt2446_with_bg_color(**d)
        # if y_idx > 5:
        #     break
    # with Pool(cpu_count()) as pool:
    #     pool.map(
    #         plot_xy_plane_specific_y_bt2446_with_bg_color_wrapper, args)

    lut = make_xyY_boundary_data2(
        color_space_name=cs.BT2020, white=cs.D65, y_num=Y_NUM, h_num=H_NUM)

    temp = tpg.equal_devision(512, 10)
    idx_list = [0] + [np.sum(temp[:x+1]) for x in range(len(temp))]

    for idx in range(len(idx_list) - 1):
        args = []
        base_list = y_val_list[idx_list[idx]:idx_list[idx+1]]
        for y_idx, y_val in enumerate(base_list):
            d = dict(y_idx=y_idx+idx_list[idx], y_val=y_val, lut=lut)
            args.append(d)
            # plot_xyY_color_volume_seq_tm(**d)
        with Pool(cpu_count()) as pool:
            pool.map(plot_xyY_color_volume_seq_tm_wrapper, args)


def plot_xyY_color_volume_seq(y_idx, lut):
    graph_name = "/work/overuse/2020/022_bt2446/xy_plane/"\
        + f"3d_sdr_no_{y_idx:04d}.png"

    x = lut[:y_idx+1, :, 0]
    y = lut[:y_idx+1, :, 1]
    buf = []
    for idx in range(y_idx + 1):
        buf.append(np.ones(H_NUM) * idx / (Y_NUM - 1))
    ly = np.vstack((buf)).reshape(x.shape)
    large_xyz = xyY_to_XYZ(np.dstack((x, y, ly)))
    print(large_xyz.shape)
    rgb = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65, BT2020_COLOURSPACE.XYZ_to_RGB_matrix)
    rgb = np.clip(rgb, 0.0, 1.0) ** (1/2.4)
    rgb = rgb.reshape(((y_idx + 1) * H_NUM, 3))

    fig = plt.figure(figsize=(9, 9))
    ax = Axes3D(fig)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Y")
    large_y = y_idx / (Y_NUM - 1)
    ax.set_title(
        f"SDR xyY Color Volume, Y = {large_y * 100:.02f} cd/m2",
        fontsize=18)
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(0.0, 0.9)
    ax.set_zlim(0.0, 1.1)
    ax.view_init(elev=20, azim=-120)
    ax.scatter(x.flatten(), y.flatten(), ly.flatten(),
               marker='o', c=rgb, zorder=1)
    plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def plot_xyY_color_volume_seq_tm(y_idx, y_val, lut):
    graph_name = "/work/overuse/2020/022_bt2446/xy_plane/"\
        + f"3d_hdr_tm_no_{y_idx:04d}.png"

    x = lut[:y_idx+1, :, 0]
    y = lut[:y_idx+1, :, 1]
    buf = []
    for idx in range(y_idx + 1):
        buf.append(np.ones(H_NUM) * idx / (Y_NUM - 1))
    ly = np.vstack((buf)).reshape(x.shape)
    large_xyz = xyY_to_XYZ(np.dstack((x, y, ly)))
    print(large_xyz.shape)
    rgb = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65, BT2020_COLOURSPACE.XYZ_to_RGB_matrix)
    rgb = np.clip(rgb, 0.0, 1.0) ** (1/2.4)
    rgb = rgb.reshape(((y_idx + 1) * H_NUM, 3))

    fig = plt.figure(figsize=(9, 9))
    ax = Axes3D(fig)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Y")
    large_y = y_val
    ax.set_title(
        f"After TM xyY Color Volume, Y = {large_y * 100:.02f} cd/m2",
        fontsize=18)
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(0.0, 0.9)
    ax.set_zlim(0.0, 1.3)
    ax.view_init(elev=20, azim=-120)
    ax.scatter(x.flatten(), y.flatten(), ly.flatten(),
               marker='o', c=rgb, zorder=1)
    plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def main_func():
    # make_xyY_boundary_data(
    #     color_space_name=cs.BT2020, white=cs.D65, y_num=Y_NUM, h_num=H_NUM)
    # plot_xy_plane(y_num=Y_NUM, h_num=H_NUM)
    plot_xy_plane_with_bt2446()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
