# -*- coding: utf-8 -*-
"""
"""

# import standard libraries
import os
import sys
from tkinter import Y

# import third-party libraries
import numpy as np
from PySide2.QtWidgets import QApplication, QHBoxLayout, QWidget, QSlider,\
    QLabel, QVBoxLayout, QGridLayout
from PySide2.QtCore import Qt
from PySide2.QtGui import QPalette, QColor
from PySide2 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg\
    import FigureCanvasQTAgg as FigureCanvas
from colour import SpectralShape
import matplotlib.pyplot as plt

# import my libraries
import color_space as cs
import plot_utility as pu
from spectrum import CIE1931_CMFS, DisplaySpectrum, create_display_sd,\
    START_WAVELENGTH, STOP_WAVELENGTH, WAVELENGTH_STEP

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


class WindowColorControl():
    def __init__(self, parent) -> None:
        self.parent = parent

    def set_bg_color(self, color=[0.15, 0.15, 0.15]):
        palette = self.parent.palette()
        color_value = [int(x * 255) for x in color]
        palette.setColor(QPalette.Window, QColor(*color_value))
        self.parent.setPalette(palette)
        self.parent.setAutoFillBackground(True)


class TyBasicSlider(QWidget):
    def __init__(
            self, int_float_rate=100, default=1.0, min_val=0.0, max_val=1.0,
            orientation=Qt.Horizontal):
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
        orientation: 
        Example
        -------
        >>> slider = TyBasicSlider(
        ...     int_float_rate=10, default=2.2, min_val=0.1, max_val=3.0)
        """
        super().__init__()
        self.int_float_rate = int_float_rate
        self.defalut_value = default
        self.slider = QSlider(orientation=orientation)
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


class ColorDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.widget = QtWidgets.QColorDialog()
        self.widget.setWindowFlags(QtCore.Qt.Widget)
        self.widget.setOptions(
            QtWidgets.QColorDialog.DontUseNativeDialog |
            QtWidgets.QColorDialog.NoButtons)

    def get_widget(self):
        return self.widget

        # layout = QtWidgets.QVBoxLayout(self)
        # layout.addWidget(widget)
        # hbox = QtWidgets.QHBoxLayout()
        # hbox.addWidget(QtWidgets.QPushButton('No Color'))
        # hbox.addWidget(QtWidgets.QPushButton('Cancel'))
        # hbox.addWidget(QtWidgets.QPushButton('Ok'))
        # layout.addLayout(hbox)


class XXXX_Objects():
    """
    Manage GUI objects related to SPD plot.
    """
    def __init__(self):
        self.create_objects()

    def create_objects(self):
        # object for widget
        self.slider = TyBasicSlider(
            int_float_rate=100, default=1.0, min_val=0.0, max_val=1.0,
            orientation=Qt.Vertical)
        self.color_dialog = ColorDialog()


class LayoutControl():
    def __init__(self, parent) -> None:
        self.base_layout = QGridLayout()
        parent.setLayout(self.base_layout)

    def set_xxxx_layout(self, xxxx_objects: XXXX_Objects):
        xxxx_layout = QHBoxLayout()
        xxxx_layout.addWidget(xxxx_objects.slider.get_widget())
        xxxx_layout.addWidget(xxxx_objects.color_dialog.get_widget())
        self.base_layout.addLayout(xxxx_layout, 0, 0)


class EventControl():
    def __init__(self) -> None:
        pass

    def set_display_sd_slider_event(self, xxxx_objects: XXXX_Objects):
        self.xxxx_objects = xxxx_objects
        self.xxxx_objects.slider.set_slot(self.print_value)

    def print_value(self):
        value = self.xxxx_objects.slider.get_value()
        print(value)


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(1080, 1080)

        # background color
        window_color = WindowColorControl(parent=self)
        window_color.set_bg_color(color=[0.8, 0.8, 0.8])

        xxxx_objects = XXXX_Objects()

        # layout
        layout = LayoutControl(self)
        layout.set_xxxx_layout(xxxx_objects=xxxx_objects)

        # set slot
        self.event_control = EventControl()
        self.event_control.set_display_sd_slider_event(
            xxxx_objects=xxxx_objects)


def main_func():
    app = QApplication([])
    widget = MyWidget()
    widget.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
