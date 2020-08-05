# -*- coding: utf-8 -*-
"""
Title
==============

Description.

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import XYZ_to_RGB, xyY_to_XYZ, RGB_COLOURSPACES,\
    xyY_to_XYZ, XYZ_to_RGB
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    inner = calc_inner_product_from_ndarray(data=data)

    cos_val = inner / (norm[:-1] * norm[1:])

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


def reduce_xyY_sample(xyY_data, reduced_sample=10):
    reduced_xyY_data = np.zeros((len(xyY_data), reduced_sample, 2))
    for idx in range(len(xyY_data)):
        data = xyY_data[idx].copy()
        data = add_data_to_start_and_end_for_inner_product(data)
        angle = calc_angle_from_ndarray(data)
        angle[angle < 1] = 180
        rd_idx = np.argsort(angle)[:reduced_sample]
        reduced_xyY_data[idx] = xyY_data[idx, rd_idx]

    return reduced_xyY_data


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
        xyY_data, xyY_org, color_space_name=cs.BT2020):
    # graph_name = "/work/overuse/2020/022_bt2446/xy_plane/"\
    #     + f"3d_sdr_no_{y_idx:04d}.png"

    x = xyY_data[..., 0].flatten()
    y = xyY_data[..., 1].flatten()
    buf = []
    for idx in range(len(xyY_data)):
        buf.append(np.ones(xyY_data.shape[1]) * idx / (len(xyY_data) - 1))
    z = np.vstack(buf).flatten()

    large_xyz = xyY_to_XYZ(np.dstack((x, y, z)))
    rgb = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65,
        RGB_COLOURSPACES[color_space_name].XYZ_to_RGB_matrix)
    rgb = np.clip(rgb, 0.0, 1.0) ** (1/2.4)
    rgb = rgb.reshape((len(x), 3))

    fig = plt.figure(figsize=(9, 9))
    ax = Axes3D(fig)
    plt.gca().patch.set_facecolor('white')
    ax.w_xaxis.set_pane_color((0.6, 0.6, 0.6, 1.0))
    ax.w_yaxis.set_pane_color((0.6, 0.6, 0.6, 1.0))
    ax.w_zaxis.set_pane_color((0.6, 0.6, 0.6, 1.0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Y")
    # ax.set_title(
    #     f"SDR xyY Color Volume, Y = {large_y * 100:.02f} cd/m2",
    #     fontsize=18)
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(0.0, 0.9)
    ax.set_zlim(0.0, 1.1)
    ax.view_init(elev=20, azim=-120)
    ax.scatter(x, y, z, marker='o', c=rgb, zorder=1)

    # for idx in range(1, len(xyY_org), 24):
    #     x = xyY_org[idx, ..., 0].flatten()
    #     y = xyY_org[idx, ..., 1].flatten()
    #     z = (np.ones_like(x) * idx / (len(xyY_org) - 1)).flatten()

    #     ax.scatter(x, y, z, marker='o', c=rgb, s=1)
    # ax.plot(x, y, z)
    st_offset_list = np.linspace(0, 1, 40, endpoint=False)
    for st_offset in st_offset_list:
        x, y, z = extract_screw_data(
            xyY_org, y_step=1, rad_st_offset=st_offset, rad_rate=4.0)
        rgb = get_rgb_from_x_y_z(x, y, z)
        ax.scatter(x, y, z, s=1, c=rgb)
        x, y, z = extract_screw_data(
            xyY_org, y_step=1, rad_st_offset=st_offset, rad_rate=-4.0)
        rgb = get_rgb_from_x_y_z(x, y, z)
        ax.scatter(x, y, z, s=1, c=rgb)
    angle_list = np.linspace(-120, -120+360, 360, endpoint=False)
    for idx, angle in enumerate(angle_list):
        ax.view_init(elev=20, azim=angle)
        fname = f"./img_seq/angle_{idx:04d}.png"
        print(fname)
        plt.savefig(
            fname, bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close(fig)


def get_rgb_from_x_y_z(x, y, z, color_space_name=cs.BT2020):
    large_xyz = xyY_to_XYZ(np.dstack((x, y, z)))
    rgb = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65,
        RGB_COLOURSPACES[color_space_name].XYZ_to_RGB_matrix)
    rgb = np.clip(rgb, 0.0, 1.0) ** (1/2.4)
    rgb = rgb.reshape((len(x), 3))

    return rgb


def extract_screw_data(xyY_data, y_step=1, rad_st_offset=0.0, rad_rate=1.5):
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


def experimental_func():
    # data = np.array([
    #     [0, 0], [1, np.sqrt(3)], [0, np.sqrt(3)], [-1, 0]])
    # data = add_data_to_start_and_end_for_inner_product(data)
    # print(data.shape)
    # calc_angle_from_ndarray(data=data)
    xyY_data = calc_xyY_boundary_data(
        color_space_name=cs.BT2020, y_num=513, h_num=1024)
    # plot_simple_xy_plane(xyY_data)
    reduced_xyY_data = reduce_xyY_sample(xyY_data=xyY_data, reduced_sample=6)
    plot_xyY_color_volume(xyY_data=reduced_xyY_data, xyY_org=xyY_data)
    # extract_screw_data(xyY_data, y_step=1, rad_st_offset=0.0, rad_rate=4.2)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    experimental_func()
