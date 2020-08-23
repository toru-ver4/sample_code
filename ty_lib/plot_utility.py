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
from cycler import cycler
from matplotlib import ticker
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.ticker import AutoMinorLocator
import colorsys
import matplotlib.font_manager as fm

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


def plot_3d_init(
        figsize=(9, 9),
        title="Title",
        title_font_size=18,
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
    plt.rcParams['grid.color'] = grid_color if grid_color else text_color
    fig = plt.figure(figsize=figsize)
    ax = Axes3D(fig)
    plt.gca().patch.set_facecolor(face_color)
    bg_color = plane_color
    ax.w_xaxis.set_pane_color(bg_color)
    ax.w_yaxis.set_pane_color(bg_color)
    ax.w_zaxis.set_pane_color(bg_color)
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
