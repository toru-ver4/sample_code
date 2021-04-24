# -*- coding: utf-8 -*-
"""
improve the 3dlut.
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
from PySide2.QtGui import QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg\
    import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# import my libraries
import plot_utility as pu


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


class GammaSlider(QWidget):
    def __init__(self, slot_func=None):
        super().__init__()
        self.int_float_rate = 100
        self.defalut_value = 2.2
        min_value = 0.1
        max_value = 3.0
        interval = 0.1
        self.slider = QSlider(orientation=Qt.Horizontal)
        self.slider.setTickInterval(int(interval * self.int_float_rate))
        self.slider.setMinimum(int(min_value * self.int_float_rate))
        self.slider.setMaximum(int(max_value * self.int_float_rate))
        self.slider.setValue(int(self.defalut_value * self.int_float_rate))
        self.set_slot(slot_func)

    def change_value(self):
        print("changed")
        value = self.get_value()
        print(value)

    def set_value(self, value=2.2):
        self.slider.setValue(int(value * self.int_float_rate))

    def get_value(self):
        return self.slider.value() / self.int_float_rate

    def get_slider(self):
        return self.slider

    def get_default(self):
        return self.defalut_value

    def set_slot(self, slot_func):
        self.slider.valueChanged.connect(slot_func)


class GammaLabel(QWidget):
    def __init__(self, default=2.2):
        super().__init__()
        self.label = QLabel(str(default))

    def get_label(self):
        return self.label

    def set_label(self, value):
        self.label.setText(str(value))


class MatplotlibTest():
    def __init__(
            self, figsize=(10, 8), sample_num=1024, init_gamma=2.2):
        super().__init__()
        self.fig = Figure(figsize=figsize)
        self.ax1 = self.fig.add_subplot(111)
        self.plot_init_plot(sample_num=sample_num, gamma=init_gamma)
        self.canvas = FigureCanvas(self.fig)

    def plot_init_plot(self, sample_num, gamma=2.2):
        self.x = np.linspace(0, 1, sample_num)
        self.y = self.x ** gamma
        self.plot_obj, =\
            self.ax1.plot(self.x, self.y, label=f"gamma={gamma}")

    def get_matplotlib_canvas(self):
        return self.canvas

    def update_plot(self, gamma=2.4):
        y = self.x ** gamma
        self.plot_obj.set_data(self.x, y)
        self.plot_obj.figure.canvas.draw()


class MyWidget(QtWidgets):
    def __init__(self):
        super().__init__()
        self.resize(1920, 1080)
        base_layout = QHBoxLayout(self)
        mpl_layout = QVBoxLayout(self)
        mpl_control_layout = QHBoxLayout(self)
        self.slider_obj = GammaSlider()
        self.gamma_label = GammaLabel(self.slider_obj.get_default())
        self.matplotlib = MatplotlibTest(
            init_gamma=self.slider_obj.get_default())

        self.slider_obj.set_slot(
            self.update_slider_relative_parameters)
        mpl_control_layout.addWidget(self.gamma_label.get_label())
        mpl_control_layout.addWidget(self.slider_obj.get_slider())
        mpl_layout.addWidget(self.matplotlib.get_matplotlib_canvas())
        mpl_layout.addLayout(mpl_control_layout)
        base_layout.addLayout(mpl_layout)

        picture = QPixmap("./img/sozai.png")
        self.label = QLabel(self)
        self.label.setPixmap(picture)
        base_layout.addWidget(self.label)
        self.setLayout(base_layout)

    def update_slider_relative_parameters(self):
        value = self.slider_obj.get_value()
        self.gamma_label.set_label(value)
        self.matplotlib.update_plot(gamma=value)


def main_func():
    app = QApplication([])
    widget = MyWidget()
    widget.show()
    sys.exit(app.exec_())


def matplot_qt5_test():
    app = QtWidgets.QApplication(sys.argv)
    wid = QtWidgets.QWidget()
    wid.resize(250, 150)
    grid = QtWidgets.QGridLayout(wid)
    fig = Figure(
        figsize=(7, 5), dpi=65, facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
    ax1 = fig.add_subplot(111)
    x = np.linspace(0, 1, 1024)
    y = x ** 2.4
    ax1.plot(x, y)
    canvas = FigureCanvas(fig)
    grid.addWidget(canvas, 0, 0)
    wid.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
    # matplot_qt5_test()
