# -*- coding: utf-8 -*-
"""
parameter adjustment tool for BT.2446 Method C
==============================================

"""

# import standard libraries
import os
from typing import NamedTuple
import ctypes
import platform

# import third-party libraries
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import numpy as np

# import my libraries
import bt2446_method_c as bmc
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


# definition
REF_IMG_WIDTH = 720


class KeyNames(NamedTuple):
    update: str = 'submit_button'
    curve_plot: str = 'curve_plot'


kns = KeyNames()


def make_dpi_aware():
    """
    https://github.com/PySimpleGUI/PySimpleGUI/issues/1179
    """
    if int(platform.release()) >= 8:
        ctypes.windll.shcore.SetProcessDpiAwareness(True)


def make_dummy_image(width=REF_IMG_WIDTH, height=int(REF_IMG_WIDTH / 16 * 9)):
    img = np.ones((height, width, 3)) * np.array([0.3, 0.3, 0.3])
    img_16bit = np.uint16(np.round(img * 0xFFFF))
    is_success, buffer = cv2.imencode(".png", img_16bit[..., ::-1])
    return buffer.tobytes()


def make_layout():
    control_layout = sg.Frame(
        "parameter control",
        [[sg.Submit("Update", key=kns.update)]])
    plot_layout = sg.Frame(
        "plot area",
        [[sg.Canvas(size=(640, 480), key=kns.curve_plot)]])
    left_side = sg.Column([[control_layout], [plot_layout]])

    tp_frame = sg.Frame(
        "test pattern",
        [[sg.Image(data=make_dummy_image()),
          sg.Image(data=make_dummy_image()),
          sg.Image(data=make_dummy_image())]])
    low_luminance_frame = sg.Frame(
        "low luminance",
        [[sg.Image(data=make_dummy_image()),
          sg.Image(data=make_dummy_image()),
          sg.Image(data=make_dummy_image())]])
    middle_luminance_frame = sg.Frame(
        "middle luminance",
        [[sg.Image(data=make_dummy_image()),
          sg.Image(data=make_dummy_image()),
          sg.Image(data=make_dummy_image())]])
    high_luminance_frame = sg.Frame(
        "high luminance",
        [[sg.Image(data=make_dummy_image()),
          sg.Image(data=make_dummy_image()),
          sg.Image(data=make_dummy_image())]])
    right_side = sg.Column([
        [tp_frame], [low_luminance_frame],
        [middle_luminance_frame], [high_luminance_frame]])

    return [[left_side, right_side]]


def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def plot_tome_curve(k1=0.8, k3=0.7, y_sdr_ip=60, y_hdr_ref=203):
    x_min = 1
    y_min = 1
    y_hdr_ip, y_sdr_wp, k2, k4 = bmc.calc_tonemapping_parameters(
        k1=k1, k3=k3, y_sdr_ip=y_sdr_ip, y_hdr_ref=y_hdr_ref)
    x = np.linspace(0, 10000, 1024)
    y = bmc.bt2446_method_c_tonemapping(
        x, k1=0.8, k3=0.7, y_sdr_ip=60, y_hdr_ref=203)
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(8, 6),
        graph_title="BT.2446 Method C",
        xlabel="HDR Luminance [cd/m2]",
        ylabel="SDR Luminance [cd/m2]",
        xlim=(x_min, 10000),
        ylim=(y_min, 250),
        linewidth=3,
        return_figure=True)
    pu.log_scale_settings(ax1)
    ax1.plot(x, y)

    # annotation
    bmc.draw_ip_wp_annotation(
        x_min, y_min, ax1, k1, y_hdr_ip, y_hdr_ref, y_sdr_wp, fontsize=16)

    # auxiliary line
    ax1.plot(
        [y_hdr_ip, y_hdr_ip, x_min], [y_min, k1 * y_hdr_ip, k1 * y_hdr_ip],
        'k--', lw=2, c='#555555')
    ax1.plot(
        [y_hdr_ref, y_hdr_ref, x_min], [y_min, y_sdr_wp, y_sdr_wp],
        'k--', lw=2, c='#555555')

    return fig, ax1


def main_func():
    window = sg.Window(title="Scatter Diagram", layout=make_layout(),
                       finalize=True)
    while True:
        event, values = window.read()
        print(event, values)
        if event is None:
            break
        elif event == kns.update:
            fig, ax = plot_tome_curve()
            canvas = window[kns.curve_plot].TKCanvas
            fig_agg = draw_figure(canvas, fig)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    make_dpi_aware()
    main_func()
