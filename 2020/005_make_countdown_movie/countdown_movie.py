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
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# import my libraries
import transfer_functions as tf
import test_pattern_generator2 as tpg
from composite import TextDrawer

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

NOTO_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansMonoCJKjp-Regular.otf"
NOTO_BOLD = "/usr/share/fonts/opentype/noto/NotoSansMonoCJKjp-Bold.otf"


class BackgroundImageColorParam(NamedTuple):
    transfer_function: str = tf.GAMMA24
    bg_luminance: float = 18.0
    fg_luminance: float = 90.0
    object_outline_luminance: float = 1.0
    step_ramp_code_values: list = [x * 64 for x in range(16)] + [1023]


class BackgroundImageCoodinateParam(NamedTuple):
    scaling_factor: int = 1
    width: int = 1920
    height: int = 1080
    crosscross_line_width: int = 4
    outline_width: int = 8
    ramp_pos_v_from_center: int = 360
    ramp_height: int = 216
    ramp_outline_width: int = 2
    step_ramp_font_size: float = 10
    step_ramp_font_offset_x: int = 10
    step_ramp_font_offset_y: int = 10


FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansMonoCJKjp-Regular.otf"


def convert_from_pillow_to_numpy(img):
    img = np.asarray(img) / 0xFF

    return img


class BackgroundImage():
    def __init__(
            self, color_param, coordinate_param, fname_base, dynamic_range):
        self.bit_depth = 10
        self.code_value_max = (1 << self.bit_depth) - 1

        # color settings
        self.transfer_function = color_param.transfer_function
        self.fg_color = self.convert_luminance_to_color_value(
            color_param.fg_luminance, self.transfer_function)
        self.bg_color = self.convert_luminance_to_color_value(
            color_param.bg_luminance, self.transfer_function)
        self.obj_outline_color = self.convert_luminance_to_color_value(
            color_param.object_outline_luminance, self.transfer_function)
        self.step_ramp_code_values\
            = np.array(color_param.step_ramp_code_values) / self.code_value_max

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
        self.ramp_pos_v = (param.ramp_pos_v_from_center + param.height // 2)\
            * param.scaling_factor
        self.ramp_obj_height = param.ramp_height * param.scaling_factor
        self.ramp_obj_width\
            = (1024 + param.ramp_outline_width * 2) * param.scaling_factor
        self.ramp_outline_width\
            = param.ramp_outline_width * param.scaling_factor
        self.step_ramp_pos_v\
            = ((param.height // 2 - param.ramp_pos_v_from_center
                - param.ramp_height)) * param.scaling_factor
        self.step_ramp_font_size\
            = param.step_ramp_font_size * param.scaling_factor
        self.step_ramp_font_offset_x\
            = param.step_ramp_font_offset_x * param.scaling_factor
        self.step_ramp_font_offset_y\
            = param.step_ramp_font_offset_y * param.scaling_factor

    def _debug_dump_param(self):
        for key, value in self.__dict__.items():
            print(key, ':', value)

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

    def draw_outline(self, img, fg_color, outline_width):
        tpg.draw_outline(img, fg_color, outline_width)

    def draw_ramp_pattern(self):
        ramp_obj_img = np.ones((self.ramp_obj_height, self.ramp_obj_width, 3))\
            * self.obj_outline_color

        ramp_width = self.ramp_obj_width - self.ramp_outline_width * 2
        ramp_height = self.ramp_obj_height - self.ramp_outline_width * 2
        ramp_img = tpg.gen_step_gradation(
            width=ramp_width, height=ramp_height, step_num=1025,
            bit_depth=10, color=(1.0, 1.0, 1.0), direction='h')
        ramp_img = ramp_img / self.code_value_max

        tpg.merge(ramp_obj_img, ramp_img,
                  (self.ramp_outline_width, self.ramp_outline_width))
        ramp_pos_h\
            = (self.width // 2) - (ramp_width // 2) - self.ramp_outline_width
        tpg.merge(self.img, ramp_obj_img, pos=(ramp_pos_h, self.ramp_pos_v))

    def draw_step_ramp_pattern(self):
        # 枠を含めた背景作成
        ramp_obj_img = np.ones((self.ramp_obj_height, self.ramp_obj_width, 3))\
            * self.obj_outline_color

        # 枠の内部の Step Ramp のパラメータ算出
        ramp_width = self.ramp_obj_width - self.ramp_outline_width * 2
        ramp_height = self.ramp_obj_height - self.ramp_outline_width * 2

        # Step Ramp をブロックごとに作成して、最後に hstack で結合
        buf = []
        width_list\
            = tpg.equal_devision(ramp_width, len(self.step_ramp_code_values))
        for code_value, width in zip(self.step_ramp_code_values, width_list):
            block_img = np.ones((ramp_height, width, 3)) * code_value
            # テキスト付与
            self.draw_text_into_step_ramp(block_img, code_value)
            buf.append(block_img)
        step_ramp_img = np.hstack(buf)

        # 枠を含む背景に Step Ramp を合成
        tpg.merge(ramp_obj_img, step_ramp_img,
                  (self.ramp_outline_width, self.ramp_outline_width))
        # メイン背景画像に合成
        ramp_pos_h\
            = (self.width // 2) - (ramp_width // 2) - self.ramp_outline_width
        tpg.merge(
            self.img, ramp_obj_img, pos=(ramp_pos_h, self.step_ramp_pos_v))

    def draw_text_into_step_ramp(self, block_img, code_value):
        fg_color = (1 - code_value) / 3 + 1.0 / 6.0
        print(fg_color)
        text = "{:>4d}".format(int(code_value * self.code_value_max + 0.5))
        text_drawer = TextDrawer(
            block_img, text,
            pos=(self.step_ramp_font_offset_x, self.step_ramp_font_offset_y),
            font_color=(fg_color, fg_color, fg_color, 1.0),
            font_size=self.step_ramp_font_size,
            transfer_functions=self.transfer_function,
            font_path=NOTO_REGULAR)
        text_drawer.draw()

    def make(self):
        """
        背景画像を生成する
        """
        self.img = np.ones((self.height, self.width, 3))
        self.img = self.img * self.bg_color
        self.draw_crisscross_line()
        self.draw_outline(self.img, self.fg_color, self.outline_width)
        self.draw_ramp_pattern()
        self.draw_step_ramp_pattern()

        # tpg.preview_image(self.img)

    def save(self):
        cv2.imwrite(
            self.filename, np.uint16(np.round(self.img[:, :, ::-1] * 0xFFFF)))
