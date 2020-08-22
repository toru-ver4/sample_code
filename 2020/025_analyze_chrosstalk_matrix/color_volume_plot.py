# -*- coding: utf-8 -*-
"""
Tools for plot of the color volume
==================================

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import XYZ_to_RGB, xyY_to_XYZ, RGB_COLOURSPACES
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# import my libraries
from color_volume_boundary_data import calc_xyY_boundary_data,\
    calc_xyY_boundary_data_log_scale
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


def simple_linear_xyY_plot():
    Yxy_obj = calc_xyY_boundary_data(
        color_space_name=cs.BT2020, y_num=512, h_num=1024)
    fig = plt.figure(figsize=(9, 9))
    ax = Axes3D(fig)
    plt.gca().patch.set_facecolor("#808080")
    bg_color = (0.3, 0.3, 0.3, 0.0)
    ax.w_xaxis.set_pane_color(bg_color)
    ax.w_yaxis.set_pane_color(bg_color)
    ax.w_zaxis.set_pane_color(bg_color)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Y")
    ax.set_title("Sample", fontsize=18)
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(0.0, 0.9)
    ax.set_zlim(0.0, 1.1)
    # ax.grid(False)
    # ax.grid(b=True, which='major', axis='x')
    # ax.grid(b=True, which='major', axis='y')
    # ax.grid(b=False, which='major', axis='z')

    rgb = calc_rgb_from_xyY_for_mpl(
        xyY=Yxy_obj.get_as_xyY(), color_space_name=cs.BT2020)
    x, y, z = np.dsplit(Yxy_obj.get_as_xyY(), 3)
    ax.scatter(x, y, z, s=1, c=rgb)

    ax.view_init(elev=20, azim=-120)
    fname = "./test_3d_simple.png"
    print(fname)
    plt.savefig(
        fname, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close(fig)


def experimental_func():
    simple_linear_xyY_plot()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    experimental_func()
