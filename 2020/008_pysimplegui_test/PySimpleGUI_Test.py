# -*- coding: utf-8 -*-
"""
PySimpleGUI を使ってみる
========================

以下を読んでチョット作りたくなった。

https://speakerdeck.com/okajun35/pythondedesukutotupuapuriwojian-dan-nizuo-rufang-fa

"""

# import standard libraries
import os
from typing import NamedTuple

# import third-party libraries
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import numpy as np
from colour import RGB_to_XYZ, XYZ_to_xy, xy_to_XYZ, XYZ_to_RGB
from colour.models import RGB_COLOURSPACES

# import my libraries
import transfer_functions as tf
import color_space as cs
import test_pattern_generator2 as tpg
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


class KeyNames(NamedTuple):
    browse: str = 'browse'
    eotf: str = 'transfer_characteristic'
    gamut: str = 'gamut'
    img_file_path: str = 'img_file_path'
    plot: str = 'plot_area'
    submit: str = 'submit_button'


kns = KeyNames()


def make_layout():
    transfer_characteristic_list = sg.Combo(
        values=[tf.GAMMA24, tf.ST2084, tf.HLG],
        default_value=tf.GAMMA24, key=kns.eotf)
    gamut_list = sg.Combo(
        values=[cs.BT709, cs.P3_D65, cs.BT2020],
        default_value=cs.BT709, key=kns.gamut)

    control_layout = [
        [sg.Text("File Path:"),
         sg.InputText("sample.png", size=(40, 1), key=kns.img_file_path),
         sg.FileBrowse(file_types=(
             ('Image Files', '*.png'), ('Image Files', '*.jpg'),
             ('Image Files', '*.tiff'), ('Image Files', '*.tif')))],
        [sg.Text("Transfer Characteristics:"), transfer_characteristic_list],
        [sg.Text("Gamut:"), gamut_list],
        [sg.Text("Plot Scatter Diagram"), sg.Submit("Plot", key=kns.submit)]
    ]

    plot_layout = [
        [sg.Text("CIE1931")],
        [sg.Canvas(size=(640, 480), key=kns.plot)]
    ]

    return [[sg.Column(control_layout), sg.Column(plot_layout)]]


def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def plot_chromaticity_diagram(rgb, xy, gamut_name=cs.BT709):
    xmin = 0.0
    xmax = 0.8
    ymin = 0.0
    ymax = 0.9
    cmf_xy = tpg._get_cmfs_xy()
    xlim = (min(0, xmin), max(0.8, xmax))
    ylim = (min(0, ymin), max(0.9, ymax))
    figsize_h = 8 * 1.0
    figsize_v = 9 * 1.0
    rate = 1.3
    # gamut の用意
    outer_gamut, _ = tpg.get_primaries(gamut_name)
    fig, ax1 = pu.plot_1_graph(
        fontsize=20 * rate,
        figsize=(figsize_h, figsize_v),
        graph_title="CIE1931 Chromaticity Diagram",
        xlabel=None, ylabel=None,
        legend_size=18 * rate,
        xlim=xlim, ylim=ylim,
        xtick=[x * 0.1 + xmin for x in
               range(int((xlim[1] - xlim[0])/0.1) + 1)],
        ytick=[x * 0.1 + ymin for x in
               range(int((ylim[1] - ylim[0])/0.1) + 1)],
        xtick_size=17 * rate,
        ytick_size=17 * rate,
        linewidth=4 * rate,
        minor_xtick_num=2, minor_ytick_num=2,
        return_figure=True)
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=3.5*rate, label=None)
    ax1.plot((cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
             '-k', lw=3.5*rate, label=None)
    ax1.plot(outer_gamut[:, 0], outer_gamut[:, 1],
             c=(0, 0, 0), label=gamut_name, lw=2.75*rate)
    ax1.plot(tpg.D65_WHITE[0], tpg.D65_WHITE[1], marker='x', c='k',
             lw=2.75*rate, label='D65', ms=10*rate, mew=2.75*rate)
    ax1.scatter(xy[..., 0], xy[..., 1], c=rgb)

    plt.legend(loc='upper right')

    return fig, ax1


def make_scatter_data(filename, eotf, gamut):
    img = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    rate = max(1, max(img.shape[1] // 512, img.shape[0] // 512))
    img = cv2.resize(img, dsize=None, fx=1/rate, fy=1/rate,
                     interpolation=cv2.INTER_NEAREST)
    img = img / np.iinfo(img.dtype).max
    img = img[:, :, ::-1]
    img_linear = tf.eotf(img, eotf)
    large_xyz = RGB_to_XYZ(img_linear, tpg.D65_WHITE, tpg.D65_WHITE,
                           RGB_COLOURSPACES[gamut].RGB_to_XYZ_matrix)
    zero_data_idx = (large_xyz[..., 0] <= 0) & (large_xyz[..., 1] <= 0)\
        & (large_xyz[..., 2] <= 0)

    large_xyz[zero_data_idx] = [1.0, 1.0, 1.0]
    xy = XYZ_to_xy(large_xyz)
    rgb = XYZ_to_RGB(xy_to_XYZ(xy), tpg.D65_WHITE, tpg.D65_WHITE,
                     RGB_COLOURSPACES[gamut].XYZ_to_RGB_matrix)
    rgb = img.reshape((rgb.shape[0] * rgb.shape[1], rgb.shape[2]))
    div_val = np.max(rgb, axis=-1).reshape((rgb.shape[0], 1))
    div_val[div_val <= 0.0] = 1.0
    rgb = rgb / div_val
    rgb = tf.oetf(rgb, eotf)

    xy = xy.reshape((xy.shape[0] * xy.shape[1], xy.shape[2]))

    return xy, rgb


def main_func():
    window = sg.Window(title="Scatter Diagram", layout=make_layout(),
                       finalize=True)
    while True:
        event, values = window.read()
        print(event, values)
        if event is None:
            break
        elif event == kns.submit:
            filename = values[kns.img_file_path]
            eotf = values[kns.eotf]
            gamut_name = values[kns.gamut]
            xy, rgb = make_scatter_data(filename, eotf, gamut_name)
            fig, ax = plot_chromaticity_diagram(
                xy=xy, rgb=rgb, gamut_name=gamut_name)
            canvas = window[kns.plot].TKCanvas
            fig_agg = draw_figure(canvas, fig)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
