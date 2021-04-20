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
    QLabel
from PySide2.QtCore import Qt
from PySide2 import QtCore, QtWidgets

# import my libraries


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
        interval = 0.05
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


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QHBoxLayout(self)
        self.slider_obj = GammaSlider()
        self.gamma_label = GammaLabel(self.slider_obj.get_default())
        self.slider_obj.set_slot(self.update_label)
        self.layout.addWidget(self.gamma_label.get_label())
        self.layout.addWidget(self.slider_obj.get_slider())
        print("unchi")

    def print_unchi(self):
        print('unchi')

    def update_label(self):
        value = self.slider_obj.get_value()
        self.gamma_label.set_label(value)


def main_func():
    app = QApplication([])
    widget = MyWidget()
    widget.resize(1920, 1080)
    widget.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
