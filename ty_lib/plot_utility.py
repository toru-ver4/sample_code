#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# 概要
plot補助ツール群

# 参考
* [matplotlibでグラフの文字サイズを大きくする](https://goo.gl/E5fLxD)
* [Customizing matplotlib](http://matplotlib.org/users/customizing.html)

"""
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from matplotlib import ticker
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import FancyArrowPatch
import colorsys
import matplotlib.font_manager as fm

from colour.continuous import MultiSignals
from colour import MultiSpectralDistributions, SpectralShape, MSDS_CMFS,\
    SDS_ILLUMINANTS, sd_to_XYZ, XYZ_to_xy, RGB_COLOURSPACES, XYZ_to_RGB,\
    xy_to_XYZ
from colour.models import RGB_COLOURSPACE_BT709
from colour.utilities import tstack
from colour.algebra import vector_dot, normalise_maximum

from scipy.spatial import Delaunay
from scipy.ndimage import convolve

from color_space import D65 as D65_WHITE
import color_space as cs
import transfer_functions as tf

CIE1931_NAME = "cie_2_1931"
CIE1931_CMFS = MultiSpectralDistributions(MSDS_CMFS[CIE1931_NAME])

ILLUMINANT_E = SDS_ILLUMINANTS['E']

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
GRAY50 = np.array([255, 255, 255]) / 255 * 0.50
GRAY10 = np.array([255, 255, 255]) / 255 * 0.10
GRAY05 = np.array([255, 255, 255]) / 255 * 0.05
GRAY90 = np.array([255, 255, 255]) / 255 * 0.90

PLOT_FONT_NAME = "Noto Sans CJK JP"
# PLOT_FONT_NAME = "Noto Sans Mono CJK JP"
# PLOT_FONT_NAME = "DejaVu Serif"
# PLOT_FONT_NAME = "DejaVu Sans"
# PLOT_FONT_NAME = "BIZ UDPGothic"
# PLOT_FONT_NAME = "Noto Sans CJK JP"
# PLOT_FONT_NAME = "DejaVu Serif"
# PLOT_FONT_NAME = "Noto Sans CJK JP"
# PLOT_FONT_NAME = "BIZ UDGothic"
# PLOT_FONT_NAME = "Noto Sans CJK JP"
# PLOT_FONT_NAME = "Noto Sans CJK JP"
# PLOT_FONT_NAME = "BIZ UDPGothic"
# PLOT_FONT_NAME = "Noto Sans CJK JP"
# PLOT_FONT_NAME = "DejaVu Sans"
# PLOT_FONT_NAME = "Noto Sans Mono CJK JP"
# PLOT_FONT_NAME = "DejaVu Sans Mono"
# PLOT_FONT_NAME = "DejaVu Sans Mono"
# PLOT_FONT_NAME = "Noto Sans CJK JP"
# PLOT_FONT_NAME = "BIZ UDGothic"


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
        alpha=None, oetf_str=tf.GAMMA24, edgecolors=None):
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
        color2 = calc_rgb_from_xyY_for_mpl(
            xyY=xyY, color_space_name=color_space_name, oetf_str=oetf_str)
    else:
        color2 = color
    x, y, z = cs.split_tristimulus_values(xyY)
    ax.scatter3D(x, y, z, s=ms, c=color2, alpha=alpha, edgecolors=edgecolors)


class Arrow3D(FancyArrowPatch):
    # def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
    def __init__(self, x1, y1, z1, x2, y2, z2, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz1 = (x1, y1, z1)
        self._xyz2 = (x2, y2, z2)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz1
        x2, y2, z2 = self._xyz2

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


def _exist_key(key, **kwargs):
    """
    check whether key is exsit in the kwargs and kwargs[key] is not None.
    """
    is_exist = key in kwargs
    is_not_none = kwargs[key] is not None if is_exist else None

    return is_exist and is_not_none


def _set_common_parameters(fontsize, **kwargs):
    # japanese font
    # ---------------------------------------
    target_font_name = PLOT_FONT_NAME
    fonts = fm.findSystemFonts()
    for font in fonts:
        font_name = fm.FontProperties(fname=font).get_name()
        if font_name == target_font_name:
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

    if _exist_key('tick_size', **kwargs):
        plt.rcParams['xtick.labelsize'] = kwargs['tick_size']
        plt.rcParams['ytick.labelsize'] = kwargs['tick_size']

    if _exist_key('xtick_size', **kwargs):
        plt.rcParams['xtick.labelsize'] = kwargs['xtick_size']

    if _exist_key('ytick_size', **kwargs):
        plt.rcParams['ytick.labelsize'] = kwargs['ytick_size']

    if _exist_key('axis_label_size', **kwargs):
        plt.rcParams['axes.labelsize'] = kwargs['axis_label_size']

    if _exist_key('graph_title_size', **kwargs):
        plt.rcParams['axes.titlesize'] = kwargs['graph_title_size']

    if _exist_key('legend_size', **kwargs):
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
    if _exist_key('linewidth', **kwargs):
        plt.rcParams['lines.linewidth'] = kwargs['linewidth']

    if _exist_key('prop_cycle', **kwargs):
        plt.rcParams['axes.prop_cycle'] = kwargs['prop_cycle']


def plot_1_graph(fontsize=20, **kwargs):
    _set_common_parameters(fontsize=fontsize, **kwargs)

    if _exist_key('figsize', **kwargs):
        figsize = kwargs['figsize']
    else:
        figsize = (10, 8)

    if _exist_key('dpi', **kwargs):
        fig = plt.figure(figsize=figsize, dpi=kwargs['dpi'])
    else:
        fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(111)

    if _exist_key('xlim', **kwargs):
        ax1.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])

    if _exist_key('ylim', **kwargs):
        ax1.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1])

    if _exist_key('graph_title', **kwargs):
        ax1.set_title(kwargs['graph_title'])

    if _exist_key('xlabel', **kwargs):
        ax1.set_xlabel(kwargs['xlabel'])

    if _exist_key('ylabel', **kwargs):
        ax1.set_ylabel(kwargs['ylabel'])

    if _exist_key('xtick', **kwargs):
        ax1.set_xticks(kwargs['xtick'])

    if _exist_key('ytick', **kwargs):
        ax1.set_yticks(kwargs['ytick'])

    if _exist_key('minor_xtick_num', **kwargs):
        minor_locator = MultipleLocator(kwargs['minor_xtick_num'])
        ax1.xaxis.set_minor_locator(minor_locator)
        ax1.xaxis.grid(which='minor', color="#C0C0C0")
        ax1.tick_params(
            axis='x', which='minor', length=0.0, grid_linestyle='--')

    if _exist_key('minor_ytick_num', **kwargs):
        minor_locator = MultipleLocator(kwargs['minor_ytick_num'])
        ax1.yaxis.set_minor_locator(minor_locator)
        ax1.yaxis.grid(which='minor', color="#C0C0C0")
        ax1.tick_params(
            axis='y', which='minor', length=0.0, grid_linestyle='--')

    if _exist_key('bg_color', **kwargs):
        # print(f"bg_color = {kwargs['bg_color']}")
        ax1.set_facecolor(kwargs['bg_color'])
    else:
        ax1.set_facecolor((0.96, 0.96, 0.96))

    if _exist_key('return_figure', **kwargs):
        if kwargs['return_figure'] is not None:
            return fig, ax1
        else:
            return ax1
    else:
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
    ax1.set_xscale('log', base=10)
    ax1.set_yscale('log', base=10)
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


def log_sacle_settings_x_linear_y_log(
        ax, alpha_major=0.8, alpha_minor=0.2,
        major_color=0.0, major_line_width=0.85,
        minor_color=0.0, minor_line_width=0.85):
    """
    Examples
    --------
    >>> x = np.linspace(0, 1, 1024)
    >>> y = tf.eotf_to_luminance(x, tf.ST2084)
    >>> fig, ax1 = pu.plot_1_graph(
    ...     fontsize=20,
    ...     figsize=(8, 8),
    ...     bg_color=None,
    ...     graph_title="Title",
    ...     graph_title_size=None,
    ...     xlabel="X Axis Label", ylabel="Y Axis Label",
    ...     axis_label_size=None,
    ...     legend_size=17,
    ...     xlim=None,
    ...     ylim=None,
    ...     xtick=[x * 128 for x in range(8)] + [1023],
    ...     ytick=None,
    ...     xtick_size=None, ytick_size=None,
    ...     linewidth=3,
    ...     minor_xtick_num=32,
    ...     minor_ytick_num=None)
    >>> pu.log_sacle_settings_x_linear_y_log(
    ...     ax=ax1, alpha_major=0.6, alpha_minor=0.1)
    >>> ax1.plot(x*1023, y, label=tf.ST2084)
    >>> fname = "./figure/st2084_log.png"
    >>> pu.show_and_save(
        ... fig=fig, legend_loc='upper left', save_fname=fname)
    """
    ax.set_yscale("log")
    ax.tick_params(
        which='major', direction='in', top=True, right=True, length=8)
    ax.tick_params(
        which='minor', direction='in', top=True, right=True, length=4)
    major_locator = ticker.LogLocator(base=10, numticks=16)
    minor_locator = ticker.LogLocator(
        base=10, subs=[x * 0.1 for x in range(10)], numticks=16)
    ax.get_yaxis().set_major_locator(major_locator)
    ax.get_yaxis().set_minor_locator(minor_locator)
    ax.grid(
        True, "major", color=str(major_color), linestyle='-',
        linewidth=major_line_width, alpha=alpha_major, zorder=-10)
    ax.grid(
        True, "minor", color=str(minor_color), linestyle='-',
        linewidth=minor_line_width, alpha=alpha_minor, zorder=-10)


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
    # plt.gca().patch.set_facecolor(face_color)
    # ax = Axes3D(fig)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
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


def show_and_save(
        fig, legend_loc=None, save_fname=None, show=False, dpi=100,
        fontsize=None, only_graph_area=False, ncol=1):
    if legend_loc is not None:
        if fontsize is not None:
            plt.legend(fontsize=fontsize, loc=legend_loc, ncol=ncol)
        else:
            plt.legend(loc=legend_loc, ncol=ncol)

    if only_graph_area:
        plt.savefig(
            save_fname, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return

    # Adjust the position
    fig.tight_layout()

    if save_fname is not None:
        plt.savefig(save_fname, dpi=dpi)

    if show:
        plt.show()

    plt.close(fig)


def draw_wl_annotation(
        ax1, wl, st_pos=[0, 0], ed_pos=[1, 1], rate=1.0):
    """
    Draw annotation of the wavelength.
    """
    arrowstyle = '-'
    linestyle = '-'
    ax1.plot(ed_pos[0], ed_pos[1], 'ko', ms=4*rate, zorder=45)
    arrowprops = dict(
        facecolor=[0, 0, 0],
        edgecolor=[0, 0, 0],
        arrowstyle=arrowstyle, linestyle=linestyle)
    ax1.annotate(
        text=f"{wl}", xy=ed_pos, xytext=st_pos, xycoords='data',
        textcoords='data', ha='center', va='center',
        arrowprops=arrowprops, fontsize=12*rate, zorder=50)


def add_first_value_to_end(data):
    """
    Examples
    --------
    >>> primaries = cs.get_primaries(color_space_name=cs.BT709)
    [[ 0.64  0.33]
     [ 0.3   0.6 ]
     [ 0.15  0.06]]
    >>> primaries2 = add_first_value_to_end(data=primaries)
    [[ 0.64  0.33]
     [ 0.3   0.6 ]
     [ 0.15  0.06]
     [ 0.64  0.33]]
    """
    new_data = np.append(data, data[0:1, :], axis=0)

    return new_data


def get_primaries(name='ITU-R BT.2020'):
    """
    Get primaries of the specific color space

    Parameters
    ----------
    name : str
        a name of the color space.

    Returns
    -------
    array_like
        prmaries. [[rx, ry], [gx, gy], [bx, by], [rx, ry]]

    """
    primaries_original = RGB_COLOURSPACES[name].primaries
    primaries = add_first_value_to_end(data=primaries_original)

    return primaries


def get_chromaticity_image(
        samples=1024, antialiasing=True, cmf_xy=None, bg_color=0.9,
        xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0):
    """
    xy色度図の馬蹄形の画像を生成する

    Returns
    -------
    ndarray
        rgb image.
    """

    """
    馬蹄の内外の判別をするために三角形で領域分割する(ドロネー図を作成)。
    ドロネー図を作れば後は外積計算で領域の内外を判別できる（たぶん）。

    なお、作成したドロネー図は以下のコードでプロット可能。
    1点補足しておくと、```plt.triplot``` の第三引数は、
    第一、第二引数から三角形を作成するための **インデックス** のリスト
    になっている。[[0, 1, 2], [2, 4, 3], ...]的な。

    ```python
    plt.figure()
    plt.triplot(xy[:, 0], xy[:, 1], triangulation.simplices.copy(), '-o')
    plt.title('triplot of Delaunay triangulation')
    plt.show()
    ```
    """
    triangulation = Delaunay(cmf_xy)

    """
    ```triangulation.find_simplex()``` で xy がどのインデックスの領域か
    調べることができる。戻り値が ```-1``` の場合は領域に含まれないため、
    0以下のリストで領域判定の mask を作ることができる。
    """
    xx, yy\
        = np.meshgrid(np.linspace(xmin, xmax, samples),
                      np.linspace(ymax, ymin, samples))
    xy = np.dstack((xx, yy))
    mask = (triangulation.find_simplex(xy) < 0).astype(np.float64)

    # アンチエイリアシングしてアルファチャンネルを滑らかに
    # ------------------------------------------------
    if antialiasing:
        kernel = np.array([
            [0, 1, 0],
            [1, 2, 1],
            [0, 1, 0],
        ]).astype(np.float64)
        kernel /= np.sum(kernel)
        mask = convolve(mask, kernel)

    # ネガポジ反転
    # --------------------------------
    mask = 1 - mask[:, :, np.newaxis]

    # xy のメッシュから色を復元
    # ------------------------
    xy[xy == 0.0] = 1.0  # ゼロ割対策
    large_xyz = xy_to_XYZ(xy)

    rgb = cs.large_xyz_to_rgb(
        xyz=large_xyz, color_space_name=cs.BT709)

    """
    そのままだとビデオレベルが低かったりするので、
    各ドット毎にRGB値を正規化＆最大化する。
    """
    rgb[rgb == 0] = 1.0  # ゼロ割対策
    rgb = normalise_maximum(rgb, axis=-1)

    # mask 適用
    # -------------------------------------
    mask_rgb = np.dstack((mask, mask, mask))
    rgb *= mask_rgb

    # 背景色をグレーに変更
    # -------------------------------------
    bg_rgb = np.ones_like(rgb)
    bg_rgb *= (1 - mask_rgb) * bg_color

    rgb += bg_rgb

    rgb = rgb ** (1/2.2)

    return rgb


def calc_horseshoe_chromaticity(st_wl=360, ed_wl=780, wl_step=1):
    spectral_shape = SpectralShape(st_wl, ed_wl, wl_step)
    wl_num = ed_wl - st_wl + wl_step
    wl = np.arange(st_wl, ed_wl + 1, wl_step)
    values = np.zeros((wl_num, wl_num))
    for idx in range(wl_num):
        values[idx, idx] = 1.0
    signals = MultiSignals(data=values, domain=wl)
    sd = MultiSpectralDistributions(data=signals)
    illuminant = ILLUMINANT_E.interpolate(shape=spectral_shape)
    cmfs = CIE1931_CMFS.trim(shape=spectral_shape)

    large_xyz = sd_to_XYZ(sd=sd, cmfs=cmfs, illuminant=illuminant)
    xy = XYZ_to_xy(large_xyz)
    add_xy = np.array([xy[0, 0], xy[0, 1]]).reshape(1, 2)
    xy = np.append(xy, add_xy, axis=0)

    return xy


def get_rotate_mtx(angle_degree=90):
    angle_rad = np.deg2rad(angle_degree)
    mtx = np.array(
        [[np.cos(angle_rad), -np.sin(angle_rad)],
         [np.sin(angle_rad), np.cos(angle_rad)]])

    return mtx


def calc_normal_pos(
        xy=np.array([[1, 3], [2, 1], [0, 0]]), normal_len=1.0,
        angle_degree=-90):
    """
    Parameters
    ----------
    xy : ndarray
        coordinate list (please see the examples.)
        shape must be (N, 2).
    """
    rotate_mtx = get_rotate_mtx(angle_degree=angle_degree)
    xy_centerd = xy[1:] - xy[:-1]
    xy_rotate = vector_dot(rotate_mtx, xy_centerd)
    aa = xy_rotate[..., 1] / xy_rotate[..., 0]
    bb = xy[:-1, 1] - xy[:-1, 0] * aa

    angle = np.arctan2(xy_rotate[..., 1], xy_rotate[..., 0])
    x4_diff = normal_len * np.cos(angle)
    x4 = xy[:-1, 0] + x4_diff
    y4 = aa * x4 + bb
    normal_pos = tstack([x4, y4])

    return normal_pos


def plot_chromaticity_diagram_base(
        rate=1.0, bt709=True, p3d65=False, bt2020=False, adobeRGB=False,
        aces_ap1=False, aces_ap0=False, d65=True):
    xmin = -0.1
    xmax = 0.8
    ymin = -0.1
    ymax = 1.0 if aces_ap0 else 0.9

    # プロット用データ準備
    # ---------------------------------
    st_wl = 380
    ed_wl = 780
    wl_step = 1
    plot_wl_list = [
        410, 450, 470, 480, 485, 490, 495,
        500, 505, 510, 520, 530, 540, 550, 560, 570, 580, 590,
        600, 620, 690]
    cmf_xy = calc_horseshoe_chromaticity(
        st_wl=st_wl, ed_wl=ed_wl, wl_step=wl_step)
    cmf_xy_norm = calc_normal_pos(
        xy=cmf_xy, normal_len=0.05, angle_degree=90)
    wl_list = np.arange(st_wl, ed_wl + 1, wl_step)
    xy_image = get_chromaticity_image(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, cmf_xy=cmf_xy)

    fig, ax1 = plot_1_graph(
        fontsize=20 * rate,
        figsize=((xmax - xmin) * 10 * rate,
                 (ymax - ymin) * 10 * rate),
        graph_title="CIE1931 Chromaticity Diagram",
        graph_title_size=None,
        xlabel=None, ylabel=None,
        axis_label_size=None,
        legend_size=14 * rate,
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
        xtick=[x * 0.1 + xmin for x in
               range(int((xmax - xmin)/0.1) + 1)],
        ytick=[x * 0.1 + ymin for x in
               range(int((ymax - ymin)/0.1) + 1)],
        xtick_size=17 * rate,
        ytick_size=17 * rate,
        linewidth=4 * rate,
        minor_xtick_num=2,
        minor_ytick_num=2)
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=2*rate, label=None)
    for idx, wl in enumerate(wl_list):
        if wl not in plot_wl_list:
            continue
        draw_wl_annotation(
            ax1=ax1, wl=wl, rate=rate,
            st_pos=[cmf_xy_norm[idx, 0], cmf_xy_norm[idx, 1]],
            ed_pos=[cmf_xy[idx, 0], cmf_xy[idx, 1]])

    ax1.imshow(xy_image, extent=(xmin, xmax, ymin, ymax), alpha=0.5)

    def draw_gamut(rate, name=cs.BT709, label="BT.709", color=RED, lw=2.75):
        gamut = get_primaries(name=name)
        ax1.plot(gamut[:, 0], gamut[:, 1], c=color, label=label, lw=lw*rate)

    if bt709:
        draw_gamut(rate=rate, name=cs.BT709, label="BT.709", color=RED)
    if adobeRGB:
        draw_gamut(rate=rate, name=cs.ADOBE_RGB, label="Adobe RGB", color=SKY)
    if p3d65:
        draw_gamut(rate=rate, name=cs.P3_D65, label="DCI-P3", color=GREEN)
    if bt2020:
        draw_gamut(rate=rate, name=cs.BT2020, label="BT.2020", color=BLUE)
    if aces_ap1:
        draw_gamut(rate=rate, name=cs.ACES_AP1, label="ACES AP1", color=PINK)
    if aces_ap0:
        ap0_gamut = get_primaries(name=cs.ACES_AP0)
        ax1.plot(
            ap0_gamut[:, 0], ap0_gamut[:, 1], '--k', label="ACES AP0",
            lw=1*rate)
    if d65:
        ax1.plot(
            [0.3127], [0.3290], '+', label='D65', ms=16*rate, mew=2*rate,
            color='k', alpha=0.8)

    return fig, ax1


def plot_chromaticity_diagram_demo(
        rate=1.3, xmin=-0.1, xmax=0.8, ymin=-0.1, ymax=1.0):
    # プロット用データ準備
    # ---------------------------------
    st_wl = 380
    ed_wl = 780
    wl_step = 1
    plot_wl_list = [
        410, 450, 470, 480, 485, 490, 495,
        500, 505, 510, 520, 530, 540, 550, 560, 570, 580, 590,
        600, 620, 690]
    cmf_xy = calc_horseshoe_chromaticity(
        st_wl=st_wl, ed_wl=ed_wl, wl_step=wl_step)
    cmf_xy_norm = calc_normal_pos(
        xy=cmf_xy, normal_len=0.05, angle_degree=90)
    wl_list = np.arange(st_wl, ed_wl + 1, wl_step)
    xy_image = get_chromaticity_image(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, cmf_xy=cmf_xy)

    fig, ax1 = plot_1_graph(
        fontsize=20 * rate,
        figsize=((xmax - xmin) * 10 * rate,
                 (ymax - ymin) * 10 * rate),
        graph_title="CIE1931 Chromaticity Diagram",
        graph_title_size=None,
        xlabel=None, ylabel=None,
        axis_label_size=None,
        legend_size=14 * rate,
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
        xtick=[x * 0.1 + xmin for x in
               range(int((xmax - xmin)/0.1) + 1)],
        ytick=[x * 0.1 + ymin for x in
               range(int((ymax - ymin)/0.1) + 1)],
        xtick_size=17 * rate,
        ytick_size=17 * rate,
        linewidth=4 * rate,
        minor_xtick_num=2,
        minor_ytick_num=2)
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=2*rate, label=None)
    for idx, wl in enumerate(wl_list):
        if wl not in plot_wl_list:
            continue
        draw_wl_annotation(
            ax1=ax1, wl=wl, rate=rate,
            st_pos=[cmf_xy_norm[idx, 0], cmf_xy_norm[idx, 1]],
            ed_pos=[cmf_xy[idx, 0], cmf_xy[idx, 1]])
    bt709_gamut = get_primaries(name=cs.BT709)
    ax1.plot(bt709_gamut[:, 0], bt709_gamut[:, 1],
             c=RED, label="BT.709", lw=2.75*rate)
    bt2020_gamut = get_primaries(name=cs.BT2020)
    ax1.plot(bt2020_gamut[:, 0], bt2020_gamut[:, 1],
             c=GREEN, label="BT.2020", lw=2.75*rate)
    dci_p3_gamut = get_primaries(name=cs.P3_D65)
    ax1.plot(dci_p3_gamut[:, 0], dci_p3_gamut[:, 1],
             c=BLUE, label="DCI-P3", lw=2.75*rate)
    adoobe_rgb_gamut = get_primaries(name=cs.ADOBE_RGB)
    ax1.plot(adoobe_rgb_gamut[:, 0], adoobe_rgb_gamut[:, 1],
             c=SKY, label="AdobeRGB", lw=2.75*rate)
    ap0_gamut = get_primaries(name=cs.ACES_AP0)
    ax1.plot(ap0_gamut[:, 0], ap0_gamut[:, 1], '--k',
             label="ACES AP0", lw=1*rate)
    ax1.plot(
        [0.3127], [0.3290], 'x', label='D65', ms=12*rate, mew=2*rate,
        color='k', alpha=0.8)
    ax1.imshow(xy_image, extent=(xmin, xmax, ymin, ymax), alpha=0.5)
    show_and_save(
        fig=fig, legend_loc='upper right',
        save_fname="./chromaticity_diagram_sample.png")


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
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Title",
        graph_title_size=None,
        xlabel="X Axis Label",
        ylabel="Y Axis Label",
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
    for y, label in zip(y_list, label_list):
        ax1.plot(x, y, label=label)
    show_and_save(fig=fig, legend_loc='upper left', save_fname=None)

    plot_chromaticity_diagram_demo(
        rate=1.3, xmin=-0.1, xmax=0.8, ymin=-0.1, ymax=1.0)
    # plt.legend(loc='upper left')
    # plt.show()
    # plt.close(fig)
