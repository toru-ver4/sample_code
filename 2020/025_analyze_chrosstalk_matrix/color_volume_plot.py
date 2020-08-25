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
import matplotlib.pyplot as plt

from pyqtgraph.Qt import QtCore, QtGui

# import my libraries
from color_volume_boundary_data import calc_xyY_boundary_data
import color_space as cs
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def simple_linear_xyY_plot():
    Yxy_obj = calc_xyY_boundary_data(
        color_space_name=cs.BT2020, y_num=512, h_num=1024)
    xyY = Yxy_obj.get_as_abL()
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
    pu.plot_xyY_with_scatter3D(ax, xyY)

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
    reduced_xyY = gmb_obj.get_reduced_data_as_abL()

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

    pu.plot_xyY_with_scatter3D(ax, reduced_xyY)

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
    reduced_xyY = gmb_obj.get_reduced_data_as_abL()
    mesh_xyY = gmb_obj.get_outline_mesh_data_as_abL(
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
    pu.plot_xyY_with_scatter3D(ax, mesh_xyY)

    # outline
    pu.plot_xyY_with_scatter3D(ax, reduced_xyY)

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
    xyY = gmb_obj.get_as_abL()

    app, w = pu.pyqtgraph_plot_3d_init(
        title="Title",
        distance=1.7,
        center=(cs.D65[0], cs.D65[1], 0.5),
        elevation=30,
        angle=-120)

    pu.plot_xyY_with_gl_GLScatterPlotItem(w=w, xyY=xyY, size=0.0001)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


def simple_linear_xyY_reduced_plot_pyqtgraph(size=0.01):
    gmb_obj = calc_xyY_boundary_data(
        color_space_name=cs.BT2020, y_num=512, h_num=1024)
    reduced_xyY = gmb_obj.get_reduced_data_as_abL()

    app, w = pu.pyqtgraph_plot_3d_init(
        title="Title",
        distance=1.7,
        center=(cs.D65[0], cs.D65[1], 0.5),
        elevation=30,
        angle=-120)

    pu.plot_xyY_with_gl_GLScatterPlotItem(w=w, xyY=reduced_xyY, size=0.01)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


def simple_linear_xyY_mesh_plot_pyqtgraph(size=0.01):
    gmb_obj = calc_xyY_boundary_data(
        color_space_name=cs.BT2020, y_num=512, h_num=1024)
    reduced_xyY = gmb_obj.get_reduced_data_as_abL()
    mesh_xyY = gmb_obj.get_outline_mesh_data_as_abL()

    app, w = pu.pyqtgraph_plot_3d_init(
        title="Title",
        distance=1.7,
        center=(cs.D65[0], cs.D65[1], 0.5),
        elevation=30,
        angle=-120)

    pu.plot_xyY_with_gl_GLScatterPlotItem(w=w, xyY=mesh_xyY, size=0.001)
    pu.plot_xyY_with_gl_GLScatterPlotItem(w=w, xyY=reduced_xyY, size=0.01)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


def experimental_func():
    # simple_linear_xyY_plot()
    # simple_linear_xyY_reduced_plot()
    simple_linear_xyY_mesh_plot()
    # simple_linear_xyY_plot_pyqtgraph()
    # simple_linear_xyY_reduced_plot_pyqtgraph()
    # simple_linear_xyY_mesh_plot_pyqtgraph()
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    experimental_func()
