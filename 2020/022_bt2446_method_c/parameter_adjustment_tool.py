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
import bt2446_plot as btp
import plot_utility as pu
from key_names import KeyNames
from event_control import EventControl
from image_processing import ImageProcessing

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


# definition
REF_IMG_WIDTH = 960
PARAM_TEXT_SIZE = (32, 1)
kns = KeyNames()


def make_dpi_aware():
    """
    https://github.com/PySimpleGUI/PySimpleGUI/issues/1179
    """
    if int(platform.release()) >= 8:
        ctypes.windll.shcore.SetProcessDpiAwareness(True)


def make_dummy_image(width=REF_IMG_WIDTH, height=int(REF_IMG_WIDTH / 16 * 6)):
    img = np.ones((height, width, 3)) * np.array([0.3, 0.3, 0.3])
    img_16bit = np.uint16(np.round(img * 0xFFFF))
    is_success, buffer = cv2.imencode(".png", img_16bit[..., ::-1])
    return buffer.tobytes()


def make_control_layout_frame():
    control_layout_frame = sg.Frame(
        "parameter control",
        [[sg.Text("HDR Reference White Liminance:", size=PARAM_TEXT_SIZE,
          justification='r'),
          sg.Spin([size for size in range(100, 300)], initial_value=203,
          change_submits=True, key=kns.hdr_ref_white_spin, size=(5, 1)),
          sg.Slider(range=(100, 300), orientation='h', size=(20, 10),
          change_submits=True, key=kns.hdr_ref_white_slider,
          default_value=203)],
         [sg.Text("HDR Peak Luminance:", size=PARAM_TEXT_SIZE,
          justification='r'),
          sg.Spin([size for size in range(500, 10000)], initial_value=1000,
          change_submits=True, key=kns.hdr_peak_white_spin, size=(5, 1)),
          sg.Slider(range=(500, 10000), orientation='h', size=(20, 10),
          change_submits=True, key=kns.hdr_peak_white_slider,
          default_value=1000)],
         [sg.Text("Alpha:", size=PARAM_TEXT_SIZE, justification='r'),
          sg.Spin([size/100 for size in range(0, 33)], initial_value=0.15,
          change_submits=True, key=kns.cross_talk_alpha_spin, size=(5, 1)),
          sg.Slider(range=(0.0, 0.33), orientation='h', size=(20, 10),
          change_submits=True, key=kns.cross_talk_alpha_slider,
          default_value=0.15, resolution=0.01)],
         [sg.Text("Sigma:", size=PARAM_TEXT_SIZE, justification='r'),
          sg.Spin([size/100 for size in range(0, 100)], initial_value=0.5,
          change_submits=True, key=kns.cross_talk_sigma_spin, size=(5, 1)),
          sg.Slider(range=(0.0, 1.0), orientation='h', size=(20, 10),
          change_submits=True, key=kns.cross_talk_sigma_slider,
          default_value=0.5, resolution=0.01)],
         [sg.Text("k1:", size=PARAM_TEXT_SIZE, justification='r'),
          sg.Spin([size/100 for size in range(50, 100)], initial_value=0.8,
          change_submits=True, key=kns.k1_spin, size=(5, 1)),
          sg.Slider(range=(0.5, 1.0), orientation='h', size=(20, 10),
          change_submits=True, key=kns.k1_slider,
          default_value=0.8, resolution=0.01)],
         [sg.Text("k3:", size=PARAM_TEXT_SIZE, justification='r'),
          sg.Spin([size/100 for size in range(50, 100)], initial_value=0.7,
          change_submits=True, key=kns.k3_spin, size=(5, 1)),
          sg.Slider(range=(0.5, 1.0), orientation='h', size=(20, 10),
          change_submits=True, key=kns.k3_slider,
          default_value=0.7, resolution=0.01)],
         [sg.Text("SDR Inflection Point:", size=PARAM_TEXT_SIZE,
          justification='r'),
          sg.Spin([size/10 for size in range(200, 1000)], initial_value=58.5,
          change_submits=True, key=kns.sdr_ip_spin, size=(5, 1)),
          sg.Slider(range=(20, 100), orientation='h', size=(20, 10),
          change_submits=True, key=kns.sdr_ip_slider,
          default_value=58.5, resolution=0.1)],
         [sg.Submit("Update", key=kns.update)],
         [sg.Submit("Load Images", key=kns.load_images)]])

    return control_layout_frame


def make_layout():

    control_layout_frame = make_control_layout_frame()

    plot_layout = sg.Frame(
        "plot area",
        [[sg.Canvas(size=(640, 480), key=kns.curve_plot)]])

    information_layout = sg.Frame(
        "Information",
        [[sg.Text("Y_HDR_ip = "),
          sg.Text("0", key=kns.info_y_hdr_ip, size=(4, 1))],
         [sg.Text("Y_SDR_wp = "),
          sg.Text("0", key=kns.info_y_sdr_wp, size=(4, 1))]])

    left_side = sg.Column([
        [control_layout_frame], [plot_layout], [information_layout]])

    tp_frame = sg.Frame(
        "test pattern",
        [[sg.Image(data=make_dummy_image(), key=kns.img_tp_mapping),
          sg.Image(data=make_dummy_image(), key=kns.img_tp_raw),
          sg.Image(data=make_dummy_image(), key=kns.img_tp_luminance)]])
    low_luminance_frame = sg.Frame(
        "low luminance",
        [[sg.Image(data=make_dummy_image(), key=kns.img_low_mapping),
          sg.Image(data=make_dummy_image(), key=kns.img_low_raw),
          sg.Image(data=make_dummy_image(), key=kns.img_low_luminance)]])
    middle_luminance_frame = sg.Frame(
        "middle luminance",
        [[sg.Image(data=make_dummy_image(), key=kns.img_mid_mapping),
          sg.Image(data=make_dummy_image(), key=kns.img_mid_raw),
          sg.Image(data=make_dummy_image(), key=kns.img_mid_luminance)]])
    high_luminance_frame = sg.Frame(
        "high luminance",
        [[sg.Image(data=make_dummy_image(), key=kns.img_high_mapping),
          sg.Image(data=make_dummy_image(), key=kns.img_high_raw),
          sg.Image(data=make_dummy_image(), key=kns.img_high_luminance)]])
    right_side = sg.Column([
        [tp_frame], [low_luminance_frame],
        [middle_luminance_frame], [high_luminance_frame]])

    return [[left_side, right_side]]


def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def main_func():
    window = sg.Window(title="Scatter Diagram", layout=make_layout(),
                       finalize=True)
    # initial plot
    fig, ax, ax_lines = btp.plot_tome_curve()
    canvas = window[kns.curve_plot].TKCanvas
    fig_agg = draw_figure(canvas, fig)

    event_controller = EventControl(
        window, fig_agg, ax_lines, ImageProcessing(REF_IMG_WIDTH))
    event_handler = event_controller.get_handler()
    while True:
        event, values = window.read()
        print(event, values)
        if event is None:
            break
        else:
            event_handler[event](values)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    make_dpi_aware()
    main_func()
