# -*- coding: utf-8 -*-
"""
Countdown動画を作るクラスの定義
===============================

"""

# import standard libraries
from typing import NamedTuple

# import third-party libraries
import numpy as np
import cv2

# import my libraries
import transfer_functions as tf
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


class BackgroundImageColorParam(NamedTuple):
    transfer_function: str = tf.GAMMA24
    bg_luminance: float = 18.0
    fg_luminance: float = 90.0


class BackgroundImageCoodinateParam(NamedTuple):
    scaling_factor: int = 1
    width: int = 1920
    height: int = 1080
    crosscross_line_width: int = 4
    outline_width: int = 8


class BackgroundImage():
    def __init__(
            self, color_param, coordinate_param, fname_base, dynamic_range):
        self.bit_depth = 10

        # color settings
        self.transfer_function = color_param.transfer_function
        self.fg_color = self.convert_luminance_to_color_value(
            color_param.fg_luminance, self.transfer_function)
        self.bg_color = self.convert_luminance_to_color_value(
            color_param.bg_luminance, self.transfer_function)

        # coordinate settings
        self.set_coordinate_param(coordinate_param)

        # file io settings
        self.filename = fname_base.format(
            dynamic_range, self.width, self.height)

    def set_coordinate_param(self, param):
        self.width = param.width * param.scaling_factor
        self.height = param.height * param.scaling_factor
        self.cc_line_width = param.crosscross_line_width * param.scaling_factor
        self.outline_width = param.outline_width * param.scaling_factor

    def _debug_dump_param(self):
        print(f"transfer_function: {self.transfer_function}")
        print(f"fg_color: {self.fg_color}")
        print(f"bg_color: {self.bg_color}")
        print(f"width: {self.width}")
        print(f"height: {self.height}")
        print(f"cc_line_width: {self.cc_line_width}")
        print(f"filename: {self.filename}")

    def convert_luminance_to_color_value(self, luminance, transfer_function):
        """
        輝度[cd/m2] から 10bit code value の RGB値に変換する。
        luminance の単位は [cd/m2]。無彩色である。
        """
        code_value = self.convert_luminance_to_code_value(
            luminance, transfer_function)
        return np.array([code_value, code_value, code_value])

    def convert_luminance_to_code_value(self, luminance, transfer_function):
        """
        輝度[cd/m2] から 10bit code value に変換する。
        luminance の単位は [cd/m2]
        """
        return tf.oetf_from_luminance(luminance, transfer_function)

    def draw_crisscross_line(self):
        # H Line
        pos_v = self.height // 2 - self.cc_line_width // 2
        pt1 = (0, pos_v)
        pt2 = (self.width, pos_v)
        tpg.draw_straight_line(
            self.img, pt1, pt2, self.fg_color, self.cc_line_width)

        # V Line
        pos_h = self.width // 2 - self.cc_line_width // 2
        pt1 = (pos_h, 0)
        pt2 = (pos_h, self.height)
        tpg.draw_straight_line(
            self.img, pt1, pt2, self.fg_color, self.cc_line_width)

    def draw_outline(self):
        # upper left
        pt1 = (0, 0)
        pt2 = (self.width, 0)
        tpg.draw_straight_line(
            self.img, pt1, pt2, self.fg_color, self.outline_width)
        pt1 = (0, 0)
        pt2 = (0, self.height)
        tpg.draw_straight_line(
            self.img, pt1, pt2, self.fg_color, self.outline_width)

        # lower right
        pt1 = (self.width - self.outline_width, 0)
        pt2 = (self.width - self.outline_width, self.height)
        tpg.draw_straight_line(
            self.img, pt1, pt2, self.fg_color, self.outline_width)
        pt1 = (0, self.height - self.outline_width)
        pt2 = (self.width, self.height - self.outline_width)
        tpg.draw_straight_line(
            self.img, pt1, pt2, self.fg_color, self.outline_width)

    def make(self):
        """
        背景画像を生成する
        """
        self.img = np.ones((self.height, self.width, 3))
        self.img = self.img * self.bg_color
        self.draw_crisscross_line()
        self.draw_outline()

        # tpg.preview_image(self.img)

    def save(self):
        cv2.imwrite(
            self.filename, np.uint16(np.round(self.img[:, :, ::-1] * 0xFFFF)))
