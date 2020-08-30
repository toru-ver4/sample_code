#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# 概要
plot補助ツール群

# 参考
* [matplotlibでグラフの文字サイズを大きくする](https://goo.gl/E5fLxD)
* [Customizing matplotlib](http://matplotlib.org/users/customizing.html)

"""

import numpy as np
from matplotlib import ticker
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
import matplotlib.patches as patches
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import FancyArrowPatch
import colorsys
import matplotlib.font_manager as fm
import color_space as cs
import transfer_functions as tf

try:
    from pyqtgraph.Qt import QtGui
    import pyqtgraph.opengl as gl
    from pyqtgraph import Vector
except ImportError:
    print("RARNING: PyQtGraph is not found")


# define
# from https://jfly.uni-koeln.de/colorset/
RED = np.array([255, 75, 0]) / 255
YELLOW = np.array([255, 241, 0]) / 255
GREEN = np.array([3, 175, 122]) / 255
BLUE = np.array([0, 90, 255]) / 255
SKY = np.array([77, 196, 255]) / 255
PINK = np.array([255, 128, 130]) / 255
ORANGE = np.array([246, 170, 0]) / 255
MAJENTA = np.array([153, 0, 153]) / 255
BROWN = np.array([128, 64, 0]) / 255

cycle_num = 6
v_offset = 0.2
s = np.arange(cycle_num) / (cycle_num - 1) * (1 - v_offset) + v_offset
s = s[::-1]

r_cycle = []
g_cycle = []
b_cycle = []

for s_val in s:
    r, g, b = colorsys.hsv_to_rgb(0.0, s_val, 0.9)
    color = "#{:02X}{:02X}{:02X}".format(np.uint8(np.round(r * 0xFF)),
                                         np.uint8(np.round(g * 0xFF)),
                                         np.uint8(np.round(b * 0xFF)))
    r_cycle.append(color)
    r, g, b = colorsys.hsv_to_rgb(0.3, s_val, 0.9)
    color = "#{:02X}{:02X}{:02X}".format(np.uint8(np.round(r * 0xFF)),
                                         np.uint8(np.round(g * 0xFF)),
                                         np.uint8(np.round(b * 0xFF)))
    g_cycle.append(color)
    r, g, b = colorsys.hsv_to_rgb(0.6, s_val, 0.9)
    color = "#{:02X}{:02X}{:02X}".format(np.uint8(np.round(r * 0xFF)),
                                         np.uint8(np.round(g * 0xFF)),
                                         np.uint8(np.round(b * 0xFF)))
    b_cycle.append(color)


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


def calc_rgb_from_XYZ_for_mpl(
        XYZ, color_space_name=cs.BT709, white=cs.D65, oetf_str=tf.GAMMA24):
    """
    calc rgb from xyY for Matplotlib.
    the shape of returned rgb is (N, 3).

    Parameters
    ----------
    XYZ : ndarray
        XYZ value
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
    >>> calc_rgb_from_XYZ_for_mpl(
    ...     XYZ=xyY_to_XYZ(xyY), color_space_name=cs.BT709, white=cs.D65,
    ...     oetf_str=tf.GAMMA24)
    [[  1.00000000e+00   1.00000000e+00   1.00000000e+00]
     [  5.11402090e-01   5.11402090e-01   5.11402090e-01]
     [  9.74790473e-01   2.66456071e-07   0.00000000e+00]
     [  0.00000000e+00   9.29450257e-01   0.00000000e+00]]
    """
    array_length = calc_tristimulus_array_length(XYZ)
    rgb_linear = cs.calc_rgb_from_XYZ(
        XYZ, color_space_name=color_space_name, white=white)
    rgb_nonlinear = tf.oetf(np.clip(rgb_linear, 0.0, 1.0), oetf_str)
    rgb_reshape = rgb_nonlinear.reshape((array_length, 3))

    return rgb_reshape


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
    rgb_linear = cs.calc_rgb_from_xyY(
        xyY, color_space_name=color_space_name, white=white)
    rgb_nonlinear = tf.oetf(np.clip(rgb_linear, 0.0, 1.0), oetf_str)
    rgb_reshape = rgb_nonlinear.reshape((array_length, 3))

    return rgb_reshape


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


def plot_xyY_with_scatter3D(
        ax, xyY, ms=2, color_space_name=cs.BT709, color='rgb',
        alpha=None, oetf_str=tf.GAMMA24):
    """
    plot xyY data with ax.scatter3D.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        axis
    xyY : ndarray
        xyY data
    ms : float
        marker size.
    color_space_name : str
        the name of the target color space.
    color : str
        'rgb': rgb value.
        '#ABCDEF' : color value
    alpha : float
        alpha value.
    """
    if color == 'rgb':
        color = calc_rgb_from_xyY_for_mpl(
            xyY=xyY, color_space_name=color_space_name, oetf_str=oetf_str)
    else:
        color = color
    x, y, z = cs.split_tristimulus_values(xyY)
    ax.scatter3D(x, y, z, s=ms, c=color, alpha=alpha)


def plot_xyY_with_gl_GLScatterPlotItem(
        w, xyY, color_space_name=cs.BT709, size=0.001, color='rgb'):
    """
    plot xyY data with ax.scatter3D.

    Parameters
    ----------
    w : GLGridItem
        GLGridItem instance.
    xyY : ndarray
        xyY data
    color_space_name : str
        the name of the target color space.
    size : float
        marker size.
    """
    xyY_plot = reshape_to_Nx3(xyY)
    rgb = calc_rgb_from_xyY_for_mpl(
        xyY_plot, color_space_name=color_space_name)
    rgba = add_alpha_channel_to_rgb(rgb)
    if color != "rgb":
        rgba = np.ones_like(rgba) * np.array(color).reshape((1, 4))
    size_val = np.ones(xyY_plot.shape[0]) * size
    sp = gl.GLScatterPlotItem(
        pos=xyY_plot, size=size_val, color=rgba, pxMode=False)
    w.addItem(sp)


class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


def _set_common_parameters(fontsize, **kwargs):
    # japanese font
    # ---------------------------------------
    fonts = fm.findSystemFonts()
    for font in fonts:
        font_name = fm.FontProperties(fname=font).get_name()
        if font_name == 'Noto Sans CJK JP':
            plt.rcParams['font.family'] = font_name
            plt.rcParams["font.weight"] = 'regular'
            plt.rcParams["axes.labelweight"] = "regular"
            plt.rcParams["axes.titleweight"] = "regular"
            print("{:s} is found".format(font_name))
            break
        # if font_name == 'Noto Sans Mono CJK JP':
        #     plt.rcParams['font.family'] = font_name
        #     print("{:s} is found".format(font_name))
        #     break

    # font size
    # ---------------------------------------
    if fontsize:
        plt.rcParams["font.size"] = fontsize

    if 'tick_size' in kwargs and kwargs['tick_size']:
        plt.rcParams['xtick.labelsize'] = kwargs['tick_size']
        plt.rcParams['ytick.labelsize'] = kwargs['tick_size']

    if 'xtick_size' in kwargs and kwargs['xtick_size']:
        plt.rcParams['xtick.labelsize'] = kwargs['xtick_size']

    if 'ytick_size' in kwargs and kwargs['ytick_size']:
        plt.rcParams['ytick.labelsize'] = kwargs['ytick_size']

    if 'axis_label_size' in kwargs and kwargs['axis_label_size']:
        plt.rcParams['axes.labelsize'] = kwargs['axis_label_size']

    if 'graph_title_size' in kwargs and kwargs['graph_title_size']:
        plt.rcParams['axes.titlesize'] = kwargs['graph_title_size']

    if 'legend_size' in kwargs and kwargs['legend_size']:
        plt.rcParams['legend.fontsize'] = kwargs['legend_size']

    # plot style
    # ---------------------------------------
    if 'grid' in kwargs:
        if kwargs['grid']:
            plt.rcParams['axes.grid'] = True
        else:
            plt.rcParams['axes.grid'] = False
    else:
        plt.rcParams['axes.grid'] = True

    # line style
    # ---------------------------------------
    if 'linewidth' in kwargs and kwargs['linewidth']:
        plt.rcParams['lines.linewidth'] = kwargs['linewidth']

    if 'prop_cycle' in kwargs and kwargs['prop_cycle']:
        plt.rcParams['axes.prop_cycle'] = kwargs['prop_cycle']


def plot_1_graph(fontsize=20, **kwargs):
    _set_common_parameters(fontsize=fontsize, **kwargs)

    if 'figsize' in kwargs and kwargs['figsize']:
        figsize = kwargs['figsize']
    else:
        figsize = (10, 8)

    if 'dpi' in kwargs and kwargs['dpi']:
        fig = plt.figure(figsize=figsize, dpi=kwargs['dpi'])
    else:
        fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(111)

    if 'xlim' in kwargs and kwargs['xlim']:
        ax1.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])

    if 'ylim' in kwargs and kwargs['ylim']:
        ax1.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1])

    if 'graph_title' in kwargs and kwargs['graph_title']:
        ax1.set_title(kwargs['graph_title'])
    else:
        ax1.set_title("Title")

    if 'xlabel' in kwargs and kwargs['xlabel']:
        ax1.set_xlabel(kwargs['xlabel'])
    else:
        # ax1.set_xlabel("X Axis Label")
        pass

    if 'ylabel' in kwargs and kwargs['ylabel']:
        ax1.set_ylabel(kwargs['ylabel'])
    else:
        # ax1.set_ylabel("Y Axis Label")
        pass

    if 'xtick' in kwargs and kwargs['xtick']:
        ax1.set_xticks(kwargs['xtick'])

    if 'ytick' in kwargs and kwargs['ytick']:
        ax1.set_yticks(kwargs['ytick'])

    if 'minor_xtick_num' in kwargs and kwargs['minor_xtick_num']:
        minor_locator = AutoMinorLocator(kwargs['minor_xtick_num'])
        ax1.xaxis.set_minor_locator(minor_locator)
        ax1.xaxis.grid(which='minor', color="#808080")
        ax1.tick_params(axis='x', which='minor', length=0.0)

    if 'minor_ytick_num' in kwargs and kwargs['minor_ytick_num']:
        minor_locator = AutoMinorLocator(kwargs['minor_ytick_num'])
        ax1.yaxis.set_minor_locator(minor_locator)
        ax1.yaxis.grid(which='minor', color="#808080")
        ax1.tick_params(axis='y', which='minor', length=0.0)

    # Adjust the position
    # ------------------------------------
    fig.tight_layout()

    if 'return_figure' in kwargs and kwargs['return_figure']:
        return fig, ax1
    else:
        return ax1


def plot_1_graph_ret_figure(fontsize=20, **kwargs):
    _set_common_parameters(fontsize=fontsize, **kwargs)

    if 'figsize' in kwargs and kwargs['figsize']:
        figsize = kwargs['figsize']
    else:
        figsize = (10, 8)

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)

    if 'xlim' in kwargs and kwargs['xlim']:
        ax1.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])

    if 'ylim' in kwargs and kwargs['ylim']:
        ax1.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1])

    if 'graph_title' in kwargs and kwargs['graph_title']:
        ax1.set_title(kwargs['graph_title'])
    else:
        ax1.set_title("Title")

    if 'xlabel' in kwargs and kwargs['xlabel']:
        ax1.set_xlabel(kwargs['xlabel'])
    else:
        ax1.set_xlabel("X Axis Label")

    if 'ylabel' in kwargs and kwargs['ylabel']:
        ax1.set_ylabel(kwargs['ylabel'])
    else:
        ax1.set_ylabel("Y Axis Label")

    if 'xtick' in kwargs and kwargs['xtick']:
        ax1.set_xticks(kwargs['xtick'])

    if 'ytick' in kwargs and kwargs['ytick']:
        ax1.set_yticks(kwargs['ytick'])

    # Adjust the position
    # ------------------------------------
    fig.tight_layout()

    return fig, ax1


def _check_hsv_space():
    """
    # 概要
    Linestyle で 明度が徐々か変わるやつを作りたいんだけど、
    HSVの値がイマイチ分からないのでプロットしてみる。
    """

    h_num = 11
    s_num = 11

    h = np.arange(h_num) / (h_num - 1)
    s = np.arange(s_num) / (s_num - 1)

    f, axarr = plt.subplots(h_num, s_num, sharex='col', sharey='row',
                            figsize=(16, 16))
    for idx in range(h_num * s_num):
        h_idx = idx % h_num
        v_idx = idx // h_num
        r, g, b = colorsys.hsv_to_rgb(h[h_idx], s[v_idx], 0.9)
        color = "#{:02X}{:02X}{:02X}".format(np.uint8(np.round(r * 0xFF)),
                                             np.uint8(np.round(g * 0xFF)),
                                             np.uint8(np.round(b * 0xFF)))
        axarr[v_idx, h_idx].add_patch(
            patches.Rectangle(
                (0, 0), 1.0, 1.0, facecolor=color
            )
        )
    plt.show()


def log_scale_settings(ax1, grid_alpha=0.5, bg_color="#E0E0E0"):
    """
    https://stackoverflow.com/questions/44078409/matplotlib-semi-log-plot-minor-tick-marks-are-gone-when-range-is-large
    """
    # Log Scale
    ax1.set_xscale('log', basex=10)
    ax1.set_yscale('log', basey=10)
    ax1.tick_params(
        which='major', direction='in', top=True, right=True, length=8)
    ax1.tick_params(
        which='minor', direction='in', top=True, right=True, length=4)
    major_locator = ticker.LogLocator(base=10, numticks=16)
    minor_locator = ticker.LogLocator(
        base=10, subs=[x * 0.1 for x in range(10)], numticks=16)
    ax1.get_xaxis().set_major_locator(major_locator)
    ax1.get_xaxis().set_minor_locator(minor_locator)
    ax1.get_xaxis().set_major_locator(ticker.LogLocator())
    ax1.grid(which='both', linestyle='-', alpha=grid_alpha)
    ax1.patch.set_facecolor(bg_color)


def pyqtgraph_plot_3d_init(
        title="Title",
        distance=1.7,
        center=(0.0, 0.0, 1.0),
        elevation=30,
        angle=-120):
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.opts['distance'] = distance
    w.opts['center'] = Vector(
        center[0], center[1], center[2])
    w.opts['elevation'] = elevation
    w.opts['azimuth'] = angle
    # w.setBackgroundColor('g')
    w.setBackgroundColor([0.5, 0.5, 0.5, 1.0])

    w.show()
    w.setWindowTitle(title)
    g = gl.GLGridItem()
    w.addItem(g)

    return app, w


def plot_3d_init(
        figsize=(9, 9),
        title="Title",
        title_font_size=18,
        color_preset=None,
        face_color=(0.0, 0.0, 0.0),
        plane_color=(0.3, 0.3, 0.3, 1.0),
        text_color=(0.5, 0.5, 0.5),
        grid_color=None,
        x_label="X",
        y_label="Y",
        z_label="Z",
        xlim=None,
        ylim=None,
        zlim=None,
        xtick=None,
        ytick=None,
        ztick=None):
    if color_preset == "light":
        face_color = (0.9, 0.9, 0.9)
        plane_color = (0.8, 0.8, 0.8, 1.0)
        text_color = (0.1, 0.1, 0.1)
        grid_color = (0.9, 0.9, 0.9)
    elif color_preset == 'dark':
        face_color = (0.1, 0.1, 0.1)
        plane_color = (0.2, 0.2, 0.2, 1.0)
        text_color = (0.8, 0.8, 0.8)
        grid_color = (0.5, 0.5, 0.5)
    else:
        pass
    plt.rcParams['grid.color'] = grid_color if grid_color else text_color
    fig = plt.figure(figsize=figsize)
    plt.gca().patch.set_facecolor(face_color)
    ax = Axes3D(fig)
    ax.set_facecolor(face_color)
    ax.w_xaxis.set_pane_color(plane_color)
    ax.w_yaxis.set_pane_color(plane_color)
    ax.w_zaxis.set_pane_color(plane_color)
    ax.set_xlabel(x_label, color=text_color)
    ax.set_ylabel(y_label, color=text_color)
    ax.set_zlabel(z_label, color=text_color)
    ax.set_title(title, fontsize=title_font_size, color=text_color)
    ax.set_xticks(xtick) if xtick else None
    ax.set_yticks(ytick) if ytick else None
    ax.set_zticks(ztick) if ztick else None
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    ax.tick_params(axis='z', colors=text_color)
    ax.set_xlim(xlim) if xlim else None
    ax.set_ylim(ylim) if ylim else None
    ax.set_zlim(zlim) if zlim else None
    setattr(Axes3D, 'arrow3D', _arrow3D)

    return fig, ax


if __name__ == '__main__':
    # _check_hsv_space()

    # sample code for plot_1_graph()
    # -------------------------------
    x = np.arange(1024) / 1023
    gamma_list = [1.0, 1.2, 1.5, 1.9, 2.4, 3.0]
    label_list = ["gamma " + str(x) for x in gamma_list]
    y_list = [x ** gamma for gamma in gamma_list]
    fig, ax1 = plot_1_graph(
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
        minor_ytick_num=None,
        return_figure=True)
    for y, label in zip(y_list, label_list):
        ax1.plot(x, y, label=label)
    plt.legend(loc='upper left')
    plt.show()
    plt.close(fig)
