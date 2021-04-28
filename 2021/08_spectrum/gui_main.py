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
    QLabel, QVBoxLayout, QGridLayout
from PySide2.QtCore import QCalendar, Qt
from PySide2 import QtWidgets
from PySide2.QtGui import QPixmap, QImage, QPalette, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg\
    import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import color_space
from colour.models import RGB_COLOURSPACE_BT709

# import my libraries
import test_pattern_generator2 as tpg
import plot_utility as pu
import transfer_functions as tf
from spectrum_calculation import calc_illuminant_d_spectrum,\
    get_cie_2_1931_cmf, calc_linear_rgb_from_single_spectrum,\
    REFRECT_100P_SD

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
        self.base_layout = QGridLayout()
        parent.setLayout(self.base_layout)

    def set_mpl_layout(self, canvas, white_label, white_slider):
        white_layout = QHBoxLayout()
        white_layout.addWidget(white_label.get_widget())
        white_layout.addWidget(white_slider.get_widget())
        mpl_layout = QVBoxLayout()
        mpl_layout.addWidget(canvas.get_widget())
        mpl_layout.addLayout(white_layout)
        # white_layout.setFrameSt
        self.base_layout.addLayout(mpl_layout, 0, 0)

    def set_color_patch_layout(self, color_patch):
        patch_layout = QVBoxLayout()
        patch_layout.addWidget(color_patch.get_widget())
        self.base_layout.addLayout(patch_layout, 0, 1)


class WindowColorControl():
    def __init__(self, parent) -> None:
        self.parent = parent

    def set_bg_color(self, color=[0.15, 0.15, 0.15]):
        palette = self.parent.palette()
        color_value = [int(x * 255) for x in color]
        palette.setColor(QPalette.Window, QColor(*color_value))
        self.parent.setPalette(palette)
        self.parent.setAutoFillBackground(True)


class ColorPatchImage():
    def __init__(self, width=256, height=256, color_temp_default=6500):
        self.label = QLabel()
        self.width = width
        self.height = height
        self.image_change(color_temp=color_temp_default)

    def ndarray_to_qimage(self, img):
        """
        Parameters
        ----------
        img : ndarray(float)
            image data. data range is from 0.0 to 1.0
        """
        self.uint8_img = np.uint8(np.round(np.clip(img, 0.0, 1.0) * 255))
        height, width = self.uint8_img.shape[:2]
        self.qimg = QImage(
            self.uint8_img.data, width, height, QImage.Format_RGB888)

        return self.qimg

    def image_change(self, color_temp=6500):
        src_sd = calc_illuminant_d_spectrum(color_temp=color_temp)
        ref_sd = REFRECT_100P_SD
        cmfs = get_cie_2_1931_cmf()

        rgb_linear = calc_linear_rgb_from_single_spectrum(
            src_sd=src_sd, ref_sd=ref_sd, cmfs=cmfs,
            color_space=RGB_COLOURSPACE_BT709).reshape((1, 1, 3))
        print(f"rgb_linear={rgb_linear}")
        rgb_srgb = tf.oetf(np.clip(rgb_linear, 0.0, 1.0), tf.SRGB)
        print(f"rgb_srgb={rgb_srgb}")

        self.qimg = self.ndarray_to_qimage(
            np.ones((self.height, self.width, 3), dtype=np.uint8) * rgb_srgb)
        self.label.setPixmap(QPixmap.fromImage(self.qimg))

    def get_widget(self):
        return self.label


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
            self, white_slider, white_label, spectrum_plot, patch_img):
        self.white_slider = white_slider
        self.white_label = white_label
        self.spectrum_plot = spectrum_plot
        self.patch_img = patch_img
        self.white_slider.set_slot(self.slider_event)

    def slider_event(self):
        color_temp = self.white_slider.get_value()
        sd = calc_illuminant_d_spectrum(color_temp)
        print(f"color_temp={color_temp}")
        self.white_label.set_label(int(color_temp))
        self.spectrum_plot.update_plot(x=sd.wavelengths, y=sd.values)
        self.patch_img.image_change(color_temp=color_temp)


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(960, 540)

        # background color
        window_color = WindowColorControl(parent=self)
        window_color.set_bg_color(color=[0.7, 0.7, 0.7])

        # layout
        layout = LayoutControl(self)

        # object for widget
        white_slider = TyBasicSlider(
            int_float_rate=1/50, default=6500, min_val=3000, max_val=10000)
        white_label = TyBasicLabel(
            default=white_slider.get_default(), suffix="K")
        spectrum_plot = TySpectrumPlot(
            default_temp=white_slider.get_default(), figsize=(10, 6))
        patch_img = ColorPatchImage(
            color_temp_default=white_slider.get_default())

        # set slot
        self.event_control = EventControl()
        self.event_control.set_white_slider_event(
            white_slider=white_slider, white_label=white_label,
            spectrum_plot=spectrum_plot, patch_img=patch_img)

        # set layout
        layout.set_mpl_layout(
            canvas=spectrum_plot,
            white_label=white_label, white_slider=white_slider)
        layout.set_color_patch_layout(color_patch=patch_img)


def main_func():
    app = QApplication([])
    widget = MyWidget()
    widget.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
