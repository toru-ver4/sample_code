# -*- coding: utf-8 -*-
"""
Tools for plot of the color volume
==================================

"""

# import standard libraries
import os
import sys

# import third-party libraries
import numpy as np
from colour import XYZ_to_RGB, xyY_to_XYZ, RGB_COLOURSPACES
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from pyqtgraph import Vector
from OpenGL.GL import *
from PyQt5.QtOpenGL import QGLWidget


# import my libraries
from color_volume_boundary_data import calc_xyY_boundary_data,\
    calc_xyY_boundary_data_log_scale, split_tristimulus_values
from common import MeasureExecTime
import color_space as cs
import test_pattern_generator2 as tpg
import transfer_functions as tf
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def reshape_to_Nx3(data):
    """
    change data's shape to (N, 3).

    Examples
    --------
    >>> data = np.ones((3, 5, 3))
    >>> data2 = reshape_to_Nx3(data)
    >>> data2.shape
    (15, 3)
    """
    return np.reshape(data, (calc_tristimulus_array_length(data), 3))


def calc_tristimulus_array_length(data_array):
    """
    calcurate the array length of tristimulus ndarray.

    Parameters
    ----------
    data_array : ndarray
        tristimulus values

    Returns
    -------
    int
        length of the tristimulus array.

    Examples
    --------
    >>> data_1 = np.ones((3, 4, 3))
    >>> calc_tristimulus_array_length(data_array=data_1)
    12
    >>> data_2 = np.ones((3))
    >>> calc_tristimulus_array_length(data_array=data_2)
    1
    """
    length = 1
    for coef in data_array.shape[:-1]:
        length = length * coef

    return length


def calc_rgb_from_xyY_for_mpl(
        xyY, color_space_name=cs.BT709, white=cs.D65, oetf_str=tf.GAMMA24):
    """
    calc rgb from xyY for Matplotlib.
    the shape of returned rgb is (N, 3).

    Parameters
    ----------
    xyY : ndarray
        xyY value
    color_space_name : str
        the name of the target color space.
    white : ndarray
        white point. ex: np.array([0.3127, 0.3290])
    oetf_str : str
        oetf name.

    Returns
    -------
    ndarray
        rgb non-linear value. the shape is (N, 3).

    Examples
    --------
    >>> xyY = np.array(
    ...     [[[0.3127, 0.3290, 1.0],
    ...       [0.3127, 0.3290, 0.2]],
    ...      [[0.64, 0.33, 0.2],
    ...       [0.30, 0.60, 0.6]]])
    >>> calc_rgb_from_xyY_for_mpl(
    ...     xyY=xyY, color_space_name=cs.BT709, white=cs.D65,
    ...     oetf_str=tf.GAMMA24)
    [[  1.00000000e+00   1.00000000e+00   1.00000000e+00]
     [  5.11402090e-01   5.11402090e-01   5.11402090e-01]
     [  9.74790473e-01   2.66456071e-07   0.00000000e+00]
     [  0.00000000e+00   9.29450257e-01   0.00000000e+00]]
    """
    array_length = calc_tristimulus_array_length(xyY)
    rgb_linear = calc_rgb_from_xyY(
        xyY, color_space_name=color_space_name, white=white)
    rgb_nonlinear = tf.oetf(np.clip(rgb_linear, 0.0, 1.0), oetf_str)
    rgb_reshape = rgb_nonlinear.reshape((array_length, 3))

    return rgb_reshape


def calc_rgb_from_xyY(xyY, color_space_name=cs.BT709, white=cs.D65):
    """
    calc rgb from xyY.

    Parameters
    ----------
    xyY : ndarray
        xyY values.
    color_space_name : str
        the name of the target color space.
    white : ndarray
        white point. ex: np.array([0.3127, 0.3290])

    Returns
    -------
    ndarray
        rgb linear value (not clipped, so negative values may be present).

    Examples
    --------
    >>> xyY = np.array(
    ...     [[0.3127, 0.3290, 1.0], [0.64, 0.33, 0.2], [0.30, 0.60, 0.6]])
    >>> calc_rgb_from_xyY(
    ...     xyY=xyY, color_space_name=cs.BT709, white=cs.D65)
    [[  1.00000000e+00   1.00000000e+00   1.00000000e+00]
     [  9.40561207e-01   1.66533454e-16  -1.73472348e-17]
     [ -2.22044605e-16   8.38962916e-01  -6.93889390e-18]]
    """
    large_xyz = xyY_to_XYZ(xyY)
    rgb = XYZ_to_RGB(
        large_xyz, white, white,
        RGB_COLOURSPACES[color_space_name].XYZ_to_RGB_matrix)

    return rgb


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


def plot_xyY_with_scatter3D(ax, xyY):
    """
    plot xyY data with ax.scatter3D.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        axis
    xyY : ndarray
        xyY data
    """
    rgb = calc_rgb_from_xyY_for_mpl(
        xyY=xyY, color_space_name=cs.BT2020)
    x, y, z = split_tristimulus_values(xyY)
    ax.scatter3D(x, y, z, s=2, c=rgb)


def simple_linear_xyY_plot():
    Yxy_obj = calc_xyY_boundary_data(
        color_space_name=cs.BT2020, y_num=512, h_num=1024)
    fig, ax = pu.plot_3d_init(
        figsize=(9, 9),
        title="Title",
        title_font_size=18,
        face_color=(0.1, 0.1, 0.1),
        plane_color=(0.2, 0.2, 0.2, 1.0),
        text_color=(0.5, 0.5, 0.5),
        grid_color=(0.3, 0.3, 0.3),
        x_label="X",
        y_label="Y",
        z_label="Z",
        xlim=[0.0, 0.8],
        ylim=[0.0, 0.9],
        zlim=[0.0, 1.1])
    rgb = calc_rgb_from_xyY_for_mpl(
        xyY=Yxy_obj.get_as_xyY(), color_space_name=cs.BT2020)
    x, y, z = split_tristimulus_values(Yxy_obj.get_as_xyY())
    ax.scatter3D(x, y, z, s=1, c=rgb)

    ax.view_init(elev=20, azim=-120)
    fname = "./test_3d_simple.png"
    print(fname)
    plt.savefig(
        fname, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close(fig)


def simple_linear_xyY_reduced_plot():
    gmb_obj = calc_xyY_boundary_data(
        color_space_name=cs.BT2020, y_num=512, h_num=1024)
    reduced_xyY = gmb_obj.get_reduced_data_as_xyY()

    fig, ax = pu.plot_3d_init(
        figsize=(9, 9),
        title="Title",
        title_font_size=18,
        face_color=(0.1, 0.1, 0.1),
        plane_color=(0.2, 0.2, 0.2, 1.0),
        text_color=(0.5, 0.5, 0.5),
        grid_color=(0.3, 0.3, 0.3),
        x_label="X",
        y_label="Y",
        z_label="Z",
        xlim=[0.0, 0.8],
        ylim=[0.0, 0.9],
        zlim=[0.0, 1.1])

    rgb = calc_rgb_from_xyY_for_mpl(
        xyY=reduced_xyY, color_space_name=cs.BT2020)
    x, y, z = split_tristimulus_values(reduced_xyY)
    ax.scatter3D(x, y, z, s=1, c=rgb)

    ax.view_init(elev=20, azim=-120)
    fname = "./test_3d_reduced.png"
    print(fname)
    plt.savefig(
        fname, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close(fig)


def simple_linear_xyY_mesh_plot():
    gmb_obj = calc_xyY_boundary_data(
        color_space_name=cs.BT2020, y_num=513, h_num=1024)
    reduced_xyY = gmb_obj.get_reduced_data_as_xyY()
    mesh_xyY = gmb_obj.get_outline_mesh_data_as_xyY(
        ab_plane_div_num=40, rad_rate=4.0, l_step=4)

    fig, ax = pu.plot_3d_init(
        figsize=(9, 9),
        title="Title",
        title_font_size=18,
        face_color=(0.1, 0.1, 0.1),
        plane_color=(0.2, 0.2, 0.2, 1.0),
        text_color=(0.5, 0.5, 0.5),
        grid_color=(0.3, 0.3, 0.3),
        x_label="X",
        y_label="Y",
        z_label="Z",
        xlim=[0.0, 0.8],
        ylim=[0.0, 0.9],
        zlim=[0.0, 1.1])

    # mesh
    plot_xyY_with_scatter3D(ax, mesh_xyY)

    # outline
    plot_xyY_with_scatter3D(ax, reduced_xyY)

    ax.view_init(elev=20, azim=-120)
    fname = "./test_3d_mesh.png"
    print(fname)
    plt.savefig(
        fname, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close(fig)


def simple_linear_xyY_plot_pyqtgraph(size=0.0001):
    gmb_obj = calc_xyY_boundary_data(
        color_space_name=cs.BT2020, y_num=512, h_num=1024)
    xyY = gmb_obj.get_as_xyY()

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

    xyY = reshape_to_Nx3(xyY)
    rgb = calc_rgb_from_xyY_for_mpl(
        xyY, color_space_name=cs.BT2020)
    rgba = add_alpha_channel_to_rgb(rgb)
    size_val = np.ones(xyY.shape[0]) * size
    sp2 = gl.GLScatterPlotItem(
        pos=xyY, size=size_val, color=rgba, pxMode=False)
    w.addItem(sp2)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


def simple_linear_xyY_reduced_plot_pyqtgraph(size=0.01):
    gmb_obj = calc_xyY_boundary_data(
        color_space_name=cs.BT2020, y_num=512, h_num=1024)
    reduced_xyY = gmb_obj.get_reduced_data_as_xyY()

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

    reduced_xyY = reshape_to_Nx3(reduced_xyY)
    rgb = calc_rgb_from_xyY_for_mpl(
        reduced_xyY, color_space_name=cs.BT2020)
    rgba = add_alpha_channel_to_rgb(rgb)
    size_val = np.ones(reduced_xyY.shape[0]) * size
    sp2 = gl.GLScatterPlotItem(
        pos=reduced_xyY, size=size_val, color=rgba, pxMode=False)
    w.addItem(sp2)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


def experimental_func():
    # simple_linear_xyY_plot()
    # simple_linear_xyY_reduced_plot()
    simple_linear_xyY_mesh_plot()
    # simple_linear_xyY_plot_pyqtgraph()
    # simple_linear_xyY_reduced_plot_pyqtgraph()
    # Yxy_obj = calc_xyY_boundary_data(
    #     color_space_name=cs.BT2020, y_num=512, h_num=1024)
    # xyY = Yxy_obj.get_as_xyY()
    # ax1 = pu.plot_1_graph()
    # ax1.plot(xyY[-3, :, 0], xyY[-2, :, 1], 'o')
    # plt.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    experimental_func()
