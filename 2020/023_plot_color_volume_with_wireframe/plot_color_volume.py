# -*- coding: utf-8 -*-
"""
Title
==============

Description.

"""

# import standard libraries
import os
import sys
from multiprocessing import Pool, cpu_count

# import third-party libraries
import numpy as np
from colour import XYZ_to_RGB, xyY_to_XYZ, RGB_COLOURSPACES,\
    xyY_to_XYZ, XYZ_to_RGB
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyqtgraph.examples
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph.Vector as Vector

# import my libraries
import color_space as cs
from common import MeasureExecTime
import plot_utility as pu
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


qt_timmer_idx = 0
w = None


def add_data_to_start_and_end_for_inner_product(data):
    """
    Examples
    --------
    >>> old_data = np.array(
    ...     [[0, 0], [2, 0], [2, 2], [1, 3], [0, 2], [-1, 1]])
    >>> new_data = add_data_to_start_and_end_for_inner_product(old_data)
    >>> print(old_data)
    [[0  0] [2  0] [2  2] [1  3] [0  2] [-1  1]]
    >>> print(new_data)
    [[-1  1] [0  0] [2  0] [2  2] [1  3] [0  2] [-1  1] [ 0  0]]
    """
    temp = np.append(data[-1].reshape(1, 2), data, axis=0)
    new_data = np.append(temp, data[0].reshape(1, 2), axis=0)

    return new_data


def calc_vector_from_ndarray(data):
    """
    Examles
    -------
    >>> data = np.array(
    ...     [[0, 0], [2, 0], [2, 2], [1, 3], [0, 2], [-1, 1]])
    >>> calc_vector_from_ndarray(data)
    [[ 2  0] [ 0  2] [-1  1] [-1 -1] [-1 -1]]
    """
    return data[1:] - data[0:-1]


def calc_norm_from_ndarray(data):
    """
    Parameters
    ----------
    data : ndarray
        2-dimensional data.

    Examples
    --------
    >>> data = np.array(
    ...     [[0, 0], [2, 0], [2, 2], [1, 3], [0, 2], [-1, 1]])
    >> calc_norm_from_ndarray(data)
    [2.         2.         1.41421356 1.41421356 1.41421356]
    """
    vec = calc_vector_from_ndarray(data=data)
    norm = np.sqrt((vec[..., 0] ** 2 + vec[..., 1] ** 2))

    return norm


def calc_inner_product_from_ndarray(data):
    """
    Parameters
    ----------
    data : ndarray
        2-dimensional data.

    Examples
    --------
    >>> data = np.array(
    ...     [[0, 0], [2, 0], [2, 2], [1, 3], [0, 2], [-1, 1]])
    >> calc_inner_product(data)
    [0 2 0 2]
    """
    vec = calc_vector_from_ndarray(data=data)
    inner = vec[:-1, 0] * vec[1:, 0] + vec[:-1, 1] * vec[1:, 1]

    return inner


def calc_angle_from_ndarray(data):
    """
    Parameters
    ----------
    data : ndarray
        2-dimensional data.

    Examples
    --------
    >>> data = np.array(
    ...     [[0, 0], [2, 0], [2, 2], [1, 3], [0, 2], [-1, 1]])
    >> calc_angle_from_ndarray(data)
    [ 90.         135.          90.         179.99999879]
    """
    norm = calc_norm_from_ndarray(data=data)
    norm[norm < 10 ** -12] = 0.0
    inner = calc_inner_product_from_ndarray(data=data)

    cos_val = inner / (norm[:-1] * norm[1:])
    cos_val = np.nan_to_num(cos_val, nan=-1)

    angle = 180 - np.rad2deg(np.arccos(cos_val))

    return angle


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


def calc_xyY_boundary_data(
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
    fname = f"./lut/xyY_LUT_YxH_{y_num}x{h_num}.npy"
    if os.path.isfile(fname):
        xyY_data = np.load(fname)
        return xyY_data
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

    out_buf[0] = out_buf[1].copy()

    np.save(fname, out_buf)
    return out_buf


def calc_xyY_boundary_data_log_scale(
        color_space_name=cs.BT2020, white=cs.D65, y_num=1024, h_num=1024,
        min_exposure=-2, max_exposure=2):
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
    fname = f"./lut/xyY_LUT_{color_space_name}_"\
        + f"exp_{min_exposure}_to_{max_exposure}_"\
        + f"Log_YxH_{y_num}x{h_num}.npy"
    y_list = tpg.get_log10_x_scale(
        sample_num=y_num, min_exposure=min_exposure,
        max_exposure=max_exposure)
    if os.path.isfile(fname):
        xyY_data = np.load(fname)
        return xyY_data, y_list
    mtime = MeasureExecTime()
    mtime.start()
    out_buf = np.zeros((y_num, h_num, 2))

    for idx, large_y in enumerate(y_list):
        print(f"idx = {idx} / {y_num}")
        out_buf[idx] = make_xyY_boundary_data_specific_Y(
            large_y=large_y, color_space_name=color_space_name,
            white=white, h_num=h_num)
        mtime.lap()
    mtime.end()

    np.save(fname, out_buf)
    return out_buf, y_list


def reduce_xyY_sample(xyY_data, threshold_angle=150):
    out_buf = []
    for idx in range(len(xyY_data)):
        data = xyY_data[idx].copy()
        data = add_data_to_start_and_end_for_inner_product(data)
        angle = calc_angle_from_ndarray(data)
        angle[angle < 2] = 180
        # if idx == 0:
        #     for iii, ddd in enumerate(data):
        #         print(f"idx={iii:04d}, value={ddd}, angle={angle[iii%1024]}")
        rd_idx = (angle < threshold_angle)
        out_buf.append(xyY_data[idx, rd_idx])

    return out_buf


def plot_simple_xy_plane(xyY_data):
    idx = 128
    reduced_sample = 10
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(8, 9),
        graph_title="CIE1931 xy plane",
        graph_title_size=None,
        xlabel="x", ylabel="y",
        axis_label_size=None,
        legend_size=17,
        xlim=[0.0, 0.8],
        ylim=[0.0, 0.9],
        xtick=[x * 0.1 for x in range(9)],
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(xyY_data[idx, :, 0], xyY_data[idx, :, 1], '-', label="xy")

    data = xyY_data[128]
    data = add_data_to_start_and_end_for_inner_product(data)
    angle = calc_angle_from_ndarray(data)
    # 前処理以上データは 180°にしてしまう
    angle[angle < 1] = 180
    # 小さい値を reduced_sample 個だけ抽出
    rd_idx = np.argsort(angle)[:reduced_sample]

    ax1.plot(
        xyY_data[idx, rd_idx, 0], xyY_data[idx, rd_idx, 1],
        'ok', label="xy")

    plt.legend(loc='upper left')
    plt.show()


def plot_xyY_color_volume(
        f_idx, xyY_reduced_data, xyY_data, y_step=1,
        rad_rate=4.0, angle=-120, line_div=40, color_space_name=cs.BT2020):
    fig = plt.figure(figsize=(9, 9))
    ax = Axes3D(fig)
    plt.gca().patch.set_facecolor("#999999")
    ax.w_xaxis.set_pane_color((0.6, 0.6, 0.6, 0.0))
    ax.w_yaxis.set_pane_color((0.6, 0.6, 0.6, 0.0))
    ax.w_zaxis.set_pane_color((0.6, 0.6, 0.6, 0.0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Y")
    ax.set_title("Sample", fontsize=18)
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(0.0, 0.9)
    ax.set_zlim(0.0, 1.1)
    ax.grid(False)
    ax.grid(b=True, which='major', axis='x')
    ax.grid(b=True, which='major', axis='y')
    ax.grid(b=False, which='major', axis='z')

    for idx in range(len(xyY_reduced_data)):
        x = xyY_reduced_data[idx][:, 0].flatten()
        y = xyY_reduced_data[idx][:, 1].flatten()
        z = np.ones_like(x) * idx / (len(xyY_reduced_data) - 1)
        rgb = get_rgb_from_x_y_z(x, y, z)
        ax.scatter(x, y, z, marker='o', c=rgb, zorder=1)

    st_offset_list = np.linspace(0, 1, line_div, endpoint=False)
    for st_offset in st_offset_list:
        x, y, z = extract_screw_data(
            xyY_data, y_step=y_step, rad_st_offset=st_offset,
            rad_rate=rad_rate)
        rgb = get_rgb_from_x_y_z(x, y, z)
        ax.scatter(x, y, z, s=1, c=rgb)
        x, y, z = extract_screw_data(
            xyY_data, y_step=y_step, rad_st_offset=st_offset,
            rad_rate=-rad_rate)
        rgb = get_rgb_from_x_y_z(x, y, z)
        ax.scatter(x, y, z, s=1, c=rgb)
    ax.view_init(elev=20, azim=angle)
    fname = "/work/overuse/2020/023_color_volume/img_seq/"\
        + f"angle_{f_idx:04d}.png"
    print(fname)
    plt.savefig(
        fname, bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close(fig)


def plot_cross_outline_pyqtplot(
        line_div, xyY_data, y_step, rad_rate, alpha=1.0,
        color_space_name=cs.BT2020, min_z=0.0, max_z=1.0,
        size=0.01):
    st_offset_list = np.linspace(0, 1, line_div, endpoint=False)
    out_buf_xyz = []
    out_buf_rgb = []

    for st_offset in st_offset_list:
        x, y, z = extract_screw_data_log_scale(
            xyY_data, y_step=y_step, rad_st_offset=st_offset,
            rad_rate=rad_rate)
        rgb = get_rgb_from_x_y_z(x, y, z, color_space_name=color_space_name)
        rgba = add_alpha_channel_to_rgb(rgb, alpha=alpha)
        z = z * (max_z - min_z) + min_z
        out_buf_xyz.append(np.dstack((x, y, z)).reshape((len(x), 3)))
        out_buf_rgb.append(rgba)

        x, y, z = extract_screw_data_log_scale(
            xyY_data, y_step=y_step, rad_st_offset=st_offset,
            rad_rate=-rad_rate)
        rgb = get_rgb_from_x_y_z(x, y, z, color_space_name=color_space_name)
        rgba = add_alpha_channel_to_rgb(rgb, alpha=alpha)
        z = z * (max_z - min_z) + min_z
        out_buf_xyz.append(np.dstack((x, y, z)).reshape((len(x), 3)))
        out_buf_rgb.append(rgba)

    out_buf_xyz = np.vstack(out_buf_xyz)
    out_buf_rgb = np.vstack(out_buf_rgb)
    # out_buf_xyz[..., 0] = out_buf_xyz[..., 0] - cs.D65[0]
    # out_buf_xyz[..., 1] = out_buf_xyz[..., 1] - cs.D65[1]
    size_val = np.ones(out_buf_xyz.shape[0]) * size
    sp1 = gl.GLScatterPlotItem(
        pos=out_buf_xyz, size=size_val, color=out_buf_rgb, pxMode=False)

    return sp1


def plot_reduced_sample_pyqtplot(
        xyY_reduced_data, alpha=1.0, color_space_name=cs.BT2020,
        min_z=0.0, max_z=1.0, size=0.01):
    out_buf_xyz = []
    out_buf_rgb = []
    for idx in range(len(xyY_reduced_data)):
        x = xyY_reduced_data[idx][:, 0].flatten()
        y = xyY_reduced_data[idx][:, 1].flatten()
        z = np.ones_like(x) * idx / (len(xyY_reduced_data) - 1)
        rgb = get_rgb_from_x_y_z(x, y, z, color_space_name=color_space_name)
        rgba = add_alpha_channel_to_rgb(rgb, alpha=alpha)

        z = z * (max_z - min_z) + min_z
        out_buf_xyz.append(np.dstack((x, y, z)).reshape((len(x), 3)))
        out_buf_rgb.append(rgba)

    out_buf_xyz = np.vstack(out_buf_xyz)
    # out_buf_xyz[..., 0] = out_buf_xyz[..., 0] - cs.D65[0]
    # out_buf_xyz[..., 1] = out_buf_xyz[..., 1] - cs.D65[1]
    out_buf_rgb = np.vstack(out_buf_rgb)
    size_val = np.ones(out_buf_xyz.shape[0]) * size
    sp = gl.GLScatterPlotItem(
        pos=out_buf_xyz, size=size_val, color=out_buf_rgb, pxMode=False)

    return sp


def plot_cross_outline(
        line_div, xyY_data, y_step, rad_rate, ax, alpha=1.0,
        color_space_name=cs.BT2020, min_z=0.0, max_z=1.0):
    st_offset_list = np.linspace(0, 1, line_div, endpoint=False)
    for st_offset in st_offset_list:
        x, y, z = extract_screw_data_log_scale(
            xyY_data, y_step=y_step, rad_st_offset=st_offset,
            rad_rate=rad_rate)
        rgb = get_rgb_from_x_y_z(x, y, z, color_space_name=color_space_name)
        z = z * (max_z - min_z) + min_z
        ax.scatter(x, y, z, s=1, c=rgb, alpha=alpha)

        x, y, z = extract_screw_data_log_scale(
            xyY_data, y_step=y_step, rad_st_offset=st_offset,
            rad_rate=-rad_rate)
        rgb = get_rgb_from_x_y_z(x, y, z, color_space_name=color_space_name)
        z = z * (max_z - min_z) + min_z
        ax.scatter(x, y, z, s=1, c=rgb, alpha=alpha)


def plot_reduced_sample(
        xyY_reduced_data, ax, alpha=1.0, color_space_name=cs.BT2020,
        min_z=0.0, max_z=1.0):
    for idx in range(len(xyY_reduced_data)):
        x = xyY_reduced_data[idx][:, 0].flatten()
        y = xyY_reduced_data[idx][:, 1].flatten()
        z = np.ones_like(x) * idx / (len(xyY_reduced_data) - 1)
        rgb = get_rgb_from_x_y_z(x, y, z, color_space_name=color_space_name)

        z = z * (max_z - min_z) + min_z
        ax.scatter(x, y, z, marker='o', c=rgb, zorder=1, alpha=alpha)


def plot_xyY_color_volume_sdr_hdr(
        f_idx, xyY_data_hdr, xyY_reduced_data_hdr,
        xyY_data_sdr, xyY_reduced_data_sdr, y_step=1,
        rad_rate=4.0, angle=-120, line_div=40, color_space_name=cs.BT2020,
        min_exposure_sdr=-4, max_exposure_sdr=0,
        min_exposure_hdr=-8, max_exposure_hdr=0):
    fig = plt.figure(figsize=(9, 9))
    ax = Axes3D(fig)
    plt.gca().patch.set_facecolor("#999999")
    ax.w_xaxis.set_pane_color((0.6, 0.6, 0.6, 0.0))
    ax.w_yaxis.set_pane_color((0.6, 0.6, 0.6, 0.0))
    ax.w_zaxis.set_pane_color((0.6, 0.6, 0.6, 0.0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Y [cd/m2]")
    ax.set_title("Color Volume comparison", fontsize=18)
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(0.0, 0.9)
    ax.set_zlim(0.0, 1.1)
    ax.set_zticks(np.linspace(0, 1, max_exposure_hdr-min_exposure_hdr))
    ax.set_zticklabels(
        ['0.001', '0.01', '0.1', '1.0', '10', '100', '1000', '10000'])
    # ax.set_zscale('log')
    ax.grid(False)
    ax.grid(b=True, which='major', axis='x')
    ax.grid(b=True, which='major', axis='y')
    ax.grid(b=True, which='major', axis='z')

    # plot hdr
    plot_reduced_sample(
        xyY_reduced_data=xyY_reduced_data_hdr, ax=ax, alpha=0.15)
    plot_cross_outline(
        line_div=line_div, xyY_data=xyY_data_hdr, y_step=y_step,
        rad_rate=rad_rate, ax=ax, alpha=0.15)

    # plot sdr
    hdr_value_list = np.linspace(0, 1, max_exposure_hdr-min_exposure_hdr)
    sdr_max = hdr_value_list[-3]
    sdr_min = hdr_value_list[-3 - (max_exposure_sdr - min_exposure_sdr) + 1]
    plot_reduced_sample(
        xyY_reduced_data=xyY_reduced_data_sdr, ax=ax,
        color_space_name=cs.BT2020, min_z=sdr_min, max_z=sdr_max)
    plot_cross_outline(
        line_div=line_div//2, xyY_data=xyY_data_sdr, y_step=y_step,
        rad_rate=rad_rate, ax=ax, color_space_name=cs.BT2020,
        min_z=sdr_min, max_z=sdr_max)

    ax.view_init(elev=20, azim=angle)
    fname = "/work/overuse/2020/023_color_volume/img_seq/"\
        + f"xyY_SDR_HDR_angle_{f_idx:04d}.png"
    print(fname)
    plt.savefig(
        fname, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close(fig)


def save_each_angle(angle=-120):
    pass


def plot_xyY_color_volume_pyqtgraph(
        f_idx, xyY_data_hdr, xyY_reduced_data_hdr,
        xyY_data_sdr, xyY_reduced_data_sdr, y_step=1,
        rad_rate=4.0, angle=-120, line_div=40, color_space_name=cs.BT2020,
        min_exposure_sdr=-4, max_exposure_sdr=0,
        min_exposure_hdr=-8, max_exposure_hdr=0):
    global w
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.opts['distance'] = 1.7
    w.opts['center'] = Vector(cs.D65[0], cs.D65[1], 0.5)
    w.opts['elevation'] = 30
    w.opts['azimuth'] = -120
    w.show()
    w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')
    g = gl.GLGridItem()
    w.addItem(g)

    # HDR
    sp1 = plot_cross_outline_pyqtplot(
        line_div=line_div, xyY_data=xyY_data_hdr, y_step=y_step,
        rad_rate=rad_rate, alpha=0.5, size=0.005)
    w.addItem(sp1)
    sp2 = plot_reduced_sample_pyqtplot(
        xyY_reduced_data=xyY_reduced_data_hdr, alpha=1.0, size=0.01)
    w.addItem(sp2)

    # SDR
    hdr_value_list = np.linspace(0, 1, max_exposure_hdr-min_exposure_hdr)
    sdr_max = hdr_value_list[-3]
    sdr_min = hdr_value_list[-3 - (max_exposure_sdr - min_exposure_sdr) + 1]
    sp3 = plot_cross_outline_pyqtplot(
        line_div=line_div, xyY_data=xyY_data_sdr, y_step=y_step,
        rad_rate=rad_rate, alpha=1.0, size=0.003,
        min_z=sdr_min, max_z=sdr_max)
    w.addItem(sp3)
    sp4 = plot_reduced_sample_pyqtplot(
        xyY_reduced_data=xyY_reduced_data_sdr, alpha=1.0, size=0.008,
        min_z=sdr_min, max_z=sdr_max)
    w.addItem(sp4)

    w.initializeGL()
    angle_list = np.linspace(-120, -120+360, 360, endpoint=False)

    for idx in range(360):
        w.opts['azimuth'] = angle_list[idx]
        w.resizeGL(1024, 768)
        w.paintGL()
        fname = "/work/overuse/2020/023_color_volume/img_seq/"\
            + f'qt_{idx:04d}.png'
        print(f"saving {fname}.")
        w.grabFrameBuffer().save(fname)

    # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #     QtGui.QApplication.instance().exec_()


def get_rgb_from_x_y_z(x, y, z, color_space_name=cs.BT2020):
    large_xyz = xyY_to_XYZ(np.dstack((x, y, z)))
    rgb = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65,
        RGB_COLOURSPACES[color_space_name].XYZ_to_RGB_matrix)
    rgb = np.clip(rgb, 0.0, 1.0) ** (1/2.4)
    rgb = rgb.reshape((len(x), 3))

    return rgb


def extract_screw_data(
        xyY_data, y_step=1, rad_st_offset=0.0, rad_rate=1.5):
    large_y_num = len(xyY_data[::y_step])
    xy_sample = len(xyY_data[0])
    xy_step = tpg.equal_devision(int(large_y_num * rad_rate), xy_sample)
    x_buf = []
    y_buf = []
    z_buf = []
    xy_idx = int(rad_st_offset * xy_sample)
    xy_cnt = 0
    for idx in range(0, len(xyY_data), y_step):
        x_buf.append(xyY_data[idx, xy_idx % xy_sample, 0])
        y_buf.append(xyY_data[idx, xy_idx % xy_sample, 1])
        z_buf.append(idx / (len(xyY_data) - 1))
        xy_idx += xy_step[xy_cnt]
        xy_cnt += 1

    return np.array(x_buf), np.array(y_buf), np.array(z_buf)


def extract_screw_data_log_scale(
        xyY_data, y_step=1, rad_st_offset=0.0, rad_rate=1.5):
    large_y_num = len(xyY_data[::y_step])
    xy_sample = len(xyY_data[0])
    xy_step = tpg.equal_devision(int(large_y_num * rad_rate), xy_sample)
    x_buf = []
    y_buf = []
    z_buf = []
    xy_idx = int(rad_st_offset * xy_sample)
    xy_cnt = 0
    for idx in range(0, len(xyY_data), y_step):
        x_buf.append(xyY_data[idx, xy_idx % xy_sample, 0])
        y_buf.append(xyY_data[idx, xy_idx % xy_sample, 1])
        z_buf.append(idx / (len(xyY_data) - 1))
        xy_idx += xy_step[xy_cnt]
        xy_cnt += 1

    return np.array(x_buf), np.array(y_buf), np.array(z_buf)


def xyY_plot_angle_test(
        angle_num=360, y_step=1, rad_rate=4.0, line_div=65):
    xyY_data = calc_xyY_boundary_data(
        color_space_name=cs.BT2020, y_num=257, h_num=1024)
    reduced_xyY_data = reduce_xyY_sample(
        xyY_data=xyY_data, threshold_angle=130)

    angale_list = np.linspace(-120, -120+360, angle_num, endpoint=False)
    args = []
    for idx, angle in enumerate(angale_list):
        d = dict(
            f_idx=idx, xyY_data=xyY_data, xyY_reduced_data=reduced_xyY_data,
            y_step=y_step, rad_rate=rad_rate, angle=angle, line_div=line_div,
            color_space_name=cs.BT2020)
        # plot_xyY_color_volume(**d)
        args.append(d)
    with Pool(cpu_count()) as pool:
        pool.map(plot_xyY_color_volume_wrapper, args)


def plot_xyY_color_volume_wrapper(args):
    plot_xyY_color_volume(**args)


def create_xyY_and_reduced_data(
        color_space_name=cs.BT2020, y_num=257, h_num=1024,
        min_exposure=-4, max_exposure=0, threshold_angle=160):
    xyY_data, y_list = calc_xyY_boundary_data_log_scale(
        color_space_name=color_space_name, y_num=y_num, h_num=h_num,
        min_exposure=min_exposure, max_exposure=max_exposure)
    reduced_xyY_data = reduce_xyY_sample(
        xyY_data=xyY_data, threshold_angle=threshold_angle)

    return xyY_data, reduced_xyY_data


def xyY_plot_sdr_hdr_test(
        angle_num=360, y_step=1, rad_rate=4.0, line_div=65,
        min_exposure_sdr=-4, max_exposure_sdr=0,
        min_exposure_hdr=-8, max_exposure_hdr=0):
    xyY_data_sdr, reduced_xyY_data_sdr = create_xyY_and_reduced_data(
        color_space_name=cs.BT709, y_num=257, h_num=1024,
        min_exposure=min_exposure_sdr, max_exposure=max_exposure_sdr,
        threshold_angle=160)

    xyY_data_hdr, reduced_xyY_data_hdr = create_xyY_and_reduced_data(
        color_space_name=cs.BT2020, y_num=257, h_num=1024,
        min_exposure=min_exposure_hdr, max_exposure=max_exposure_hdr,
        threshold_angle=160)

    angale_list = np.linspace(-120, -120+360, angle_num, endpoint=False)
    args = []
    for idx, angle in enumerate(angale_list):
        d = dict(
            f_idx=idx,
            xyY_data_hdr=xyY_data_hdr,
            xyY_reduced_data_hdr=reduced_xyY_data_hdr,
            xyY_data_sdr=xyY_data_sdr,
            xyY_reduced_data_sdr=reduced_xyY_data_sdr,
            y_step=y_step, rad_rate=rad_rate, angle=angle,
            line_div=line_div, color_space_name=cs.BT2020,
            min_exposure_sdr=min_exposure_sdr,
            max_exposure_sdr=max_exposure_sdr,
            min_exposure_hdr=min_exposure_hdr,
            max_exposure_hdr=max_exposure_hdr)
        # plot_xyY_color_volume_sdr_hdr(**d)
        plot_xyY_color_volume_pyqtgraph(**d)
        args.append(d)
        break
    # with Pool(cpu_count()) as pool:
    #     pool.map(plot_xyY_color_volume_sdr_hdr_wrapper, args)


def plot_xyY_color_volume_sdr_hdr_wrapper(args):
    plot_xyY_color_volume_sdr_hdr(**args)


def add_alpha_channel_to_rgb(rgb, alpha=1.0):
    """
    Add an alpha channel to rgb array.
    This function is for pyqtgraph,
    therefore don't use for image processing.

    Example
    -------
    >>> x = np.linspace(0, 1, 10)
    >>> rgb = np.dstack((x, x, x))
    >>> add_alpha_channel_to_rgb(rgb, alpha=0.7)
    [[[ 0.          0.          0.          0.7       ]
      [ 0.11111111  0.11111111  0.11111111  0.7       ]
      [ 0.22222222  0.22222222  0.22222222  0.7       ]
      [ 0.33333333  0.33333333  0.33333333  0.7       ]
      [ 0.44444444  0.44444444  0.44444444  0.7       ]
      [ 0.55555556  0.55555556  0.55555556  0.7       ]
      [ 0.66666667  0.66666667  0.66666667  0.7       ]
      [ 0.77777778  0.77777778  0.77777778  0.7       ]
      [ 0.88888889  0.88888889  0.88888889  0.7       ]
      [ 1.          1.          1.          0.7       ]]]
    """
    after_shape = list(rgb.shape)
    after_shape[-1] = after_shape[-1] + 1
    rgba = np.dstack(
        (rgb[..., 0], rgb[..., 1], rgb[..., 2],
         np.ones_like(rgb[..., 0]) * alpha)).reshape(
        after_shape)

    return rgba


def experimental_func():
    # data = np.array([
    #     [0, 0], [1, np.sqrt(3)], [0, np.sqrt(3)], [-1, 0]])
    # data = add_data_to_start_and_end_for_inner_product(data)
    # print(data.shape)
    # calc_angle_from_ndarray(data=data)
    # data = np.array(
    #     [[0.67092612, 0.32679978], [0.6685815, 0.329], [0.6685815, 0.329]])
    # data = add_data_to_start_and_end_for_inner_product(data)
    # angle = calc_angle_from_ndarray(data)
    # print(angle)
    # xyY_data = calc_xyY_boundary_data(
    #     color_space_name=cs.BT2020, y_num=257, h_num=1024)
    # reduced_xyY_data = reduce_xyY_sample(
    #     xyY_data=xyY_data, threshold_angle=130)
    # plot_xyY_color_volume(
    #     9999, reduced_xyY_data, xyY_data, y_step=1,
    #     rad_rate=4.0, angle=-120, line_div=50, color_space_name=cs.BT2020)
    # xyY_plot_angle_test(angle_num=360, y_step=1, rad_rate=8.0, line_div=65)
    xyY_plot_sdr_hdr_test(
        angle_num=360, y_step=1, rad_rate=4.0, line_div=128,
        min_exposure_sdr=-4, max_exposure_sdr=0,
        min_exposure_hdr=-8, max_exposure_hdr=0)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    experimental_func()
    # pyqtgraph.examples.run()
    # x = np.linspace(0, 1, 10)
    # rgb = np.dstack((x, x, x)).reshape(1, 10, 3)
    # rgba = add_alpha_channel_to_rgb(rgb, alpha=0.7)
    # print(rgb.shape)
    # print(rgba)
