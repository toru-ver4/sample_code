# -*- coding: utf-8 -*-
"""
simulation
"""

# import standard libraries
import os
import sys

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt

from PySide2.QtWidgets import QApplication, QHBoxLayout, QWidget, QSlider,\
    QLabel, QVBoxLayout
from PySide2.QtCore import Qt
from PySide2 import QtWidgets
from PySide2.QtGui import QPixmap, QImage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg\
    import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# import my libraries
import test_pattern_generator2 as tpg
import plot_utility as pu
from spectrum_calculation import calc_illuminant_d_spectrum,\
    get_cie_2_1931_cmf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


class TyBasicSlider(QWidget):
    def __init__(
            self, int_float_rate=10, default=2.2, min_val=0.1, max_val=3.0):
        """
        Parameters
        ----------
        int_float_rate : float
            int to float rate.
        default : float
            default value
        min_val : float
            minimum value
        max_val : float
            maximum value

        Example
        -------
        >>> slider = TyBasicSlider(
        ...     int_float_rate=10, default=2.2, min_val=0.1, max_val=3.0)
        """
        super().__init__()
        self.int_float_rate = int_float_rate
        self.defalut_value = default
        self.slider = QSlider(orientation=Qt.Horizontal)
        self.slider.setMinimum(int(min_val * self.int_float_rate))
        self.slider.setMaximum(int(max_val * self.int_float_rate))
        self.slider.setValue(int(self.defalut_value * self.int_float_rate))
        self.slider.setTickPosition(QSlider.TicksBelow)

    def set_value(self, value=2.2):
        self.slider.setValue(int(value * self.int_float_rate))

    def get_value(self):
        return self.slider.value() / self.int_float_rate

    def get_widget(self):
        return self.slider

    def get_default(self):
        return self.defalut_value

    def set_slot(self, slot_func):
        self.slot_func = slot_func
        self.slider.valueChanged.connect(self.slot_func)


class TyBasicLabel(QWidget):
    def __init__(self, default=2.2, prefix="", suffix=""):
        super().__init__()
        self.label = QLabel(f"{prefix} {str(default)} {suffix}")
        self.prefix = prefix
        self.suffix = suffix

    def get_widget(self):
        return self.label

    def set_label(self, value):
        self.label.setText(
            f"{self.prefix} {str(value)} {self.suffix}")


class LayoutControl():
    def __init__(self, parent) -> None:
        # self.base_layout = QHBoxLayout(parent)
        self.base_layout = QHBoxLayout()
        parent.setLayout(self.base_layout)

    def set_mpl_layout(self, canvas, white_label, white_slider):
        white_layout = QHBoxLayout()
        white_layout.addWidget(white_label.get_widget())
        white_layout.addWidget(white_slider.get_widget())
        mpl_layout = QVBoxLayout()
        mpl_layout.addWidget(canvas.get_widget())
        mpl_layout.addLayout(white_layout)
        # white_layout.setFrameSt
        self.base_layout.addLayout(mpl_layout)


class TySpectrumPlot():
    def __init__(
            self, figsize=(10, 8), default_temp=6500):
        super().__init__()
        self.fig, self.ax1 = pu.plot_1_graph(
            fontsize=16,
            figsize=figsize,
            graph_title="Spectral power distribution",
            graph_title_size=None,
            xlabel="Wavelength [nm]", ylabel="???",
            axis_label_size=None,
            legend_size=14,
            xlim=[330, 860],
            ylim=[-0.05, 2.0],
            xtick=[350 + x * 50 for x in range(11)],
            ytick=None,
            xtick_size=None, ytick_size=None,
            linewidth=3,
            minor_xtick_num=None,
            minor_ytick_num=None,
            return_figure=True)
        self.plot_init_plot(sample_num=256, default_temp=6500)
        self.canvas = FigureCanvas(self.fig)
        # self.plot_obj.figure.canvas.draw()

    def plot_init_plot(self, sample_num, default_temp):
        illuminant_d = calc_illuminant_d_spectrum(color_temp=default_temp)
        self.illuminant_x = illuminant_d.wavelengths
        self.illuminant_y = illuminant_d.values
        self.illuminant_line, = self.ax1.plot(
            self.illuminant_x, self.illuminant_y, color=[0.2, 0.2, 0.2],
            label="SPD of illuminant D")

        cmf = get_cie_2_1931_cmf()
        self.cmf_x = cmf.wavelengths
        self.cmf_r = cmf.values[..., 0]
        self.cmf_g = cmf.values[..., 1]
        self.cmf_b = cmf.values[..., 2]
        self.line_cmf_r = self.ax1.plot(
            self.cmf_x, self.cmf_r, '--', color=pu.RED,
            label="Color matching function(R)")
        self.line_cmf_g = self.ax1.plot(
            self.cmf_x, self.cmf_g, '--', color=pu.GREEN,
            label="Color matching function(G)")
        self.line_cmf_b = self.ax1.plot(
            self.cmf_x, self.cmf_b, '--', color=pu.BLUE,
            label="Color matching function(B)")
        plt.legend(loc='upper right')

    def get_widget(self):
        return self.canvas

    def update_plot(self, x, y):
        self.illuminant_line.set_data(x, y)
        self.illuminant_line.figure.canvas.draw()


class EventControl():
    def __init__(self) -> None:
        pass

    def set_white_slider_event(
            self, white_slider, white_label, spectrum_plot):
        self.white_slider = white_slider
        self.white_label = white_label
        self.spectrum_plot = spectrum_plot
        self.white_slider.set_slot(self.slider_event)

    def slider_event(self):
        color_temp = self.white_slider.get_value()
        sd = calc_illuminant_d_spectrum(color_temp)
        print(f"color_temp={color_temp}")
        self.white_label.set_label(int(color_temp))
        self.spectrum_plot.update_plot(x=sd.wavelengths, y=sd.values)


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(960, 540)

        # layout
        layout = LayoutControl(self)

        # object for widget
        white_slider = TyBasicSlider(
            int_float_rate=1/50, default=6500, min_val=3000, max_val=10000)
        white_label = TyBasicLabel(
            default=white_slider.get_default(), suffix="K")
        spectrum_plot = TySpectrumPlot(
            default_temp=white_slider.get_default(), figsize=(10, 6))

        # set slot
        self.event_control = EventControl()
        self.event_control.set_white_slider_event(
            white_slider=white_slider, white_label=white_label,
            spectrum_plot=spectrum_plot)

        # set layout
        layout.set_mpl_layout(
            canvas=spectrum_plot,
            white_label=white_label, white_slider=white_slider)


def main_func():
    app = QApplication([])
    widget = MyWidget()
    widget.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
