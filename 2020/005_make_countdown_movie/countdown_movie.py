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
from font_control import TextDrawer
from font_control import NOTO_SANS_MONO_BOLD, NOTO_SANS_MONO_BLACK

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
    sound_lumiannce: float = 30.0
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
    sound_text_font_size: float = 50
    sound_text_font_path: str = NOTO_SANS_MONO_BLACK


def convert_from_pillow_to_numpy(img):
    img = np.asarray(img) / 0xFF

    return img


class BackgroundImage():
    def __init__(
            self, color_param, coordinate_param, fname_base, dynamic_range,
            scale_factor):
        self.bit_depth = 10
        self.code_value_max = (1 << self.bit_depth) - 1

        # color settings
        self.transfer_function = color_param.transfer_function
        self.fg_color = tpg.convert_luminance_to_color_value(
            color_param.fg_luminance, self.transfer_function)
        self.bg_color = tpg.convert_luminance_to_color_value(
            color_param.bg_luminance, self.transfer_function)
        self.obj_outline_color = tpg.convert_luminance_to_color_value(
            color_param.object_outline_luminance, self.transfer_function)
        self.step_ramp_code_values\
            = np.array(color_param.step_ramp_code_values) / self.code_value_max
        self.sound_text_color = tpg.convert_luminance_to_color_value(
            color_param.sound_lumiannce, self.transfer_function)

        # coordinate settings
        self.set_coordinate_param(coordinate_param, scale_factor)

        # file io settings
        self.filename = fname_base.format(
            dynamic_range, self.width, self.height)

    def set_coordinate_param(self, param, scale_factor):
        self.width = param.width * scale_factor
        self.height = param.height * scale_factor
        self.cc_line_width = param.crosscross_line_width * scale_factor
        self.outline_width = param.outline_width * scale_factor
        self.ramp_pos_v = (param.ramp_pos_v_from_center + param.height // 2)\
            * scale_factor
        self.ramp_obj_height = param.ramp_height * scale_factor
        self.ramp_obj_width\
            = (1024 + param.ramp_outline_width * 2) * scale_factor
        self.ramp_outline_width\
            = param.ramp_outline_width * scale_factor
        self.step_ramp_pos_v\
            = ((param.height // 2 - param.ramp_pos_v_from_center
                - param.ramp_height)) * scale_factor
        self.step_ramp_font_size\
            = param.step_ramp_font_size * scale_factor
        self.step_ramp_font_offset_x\
            = param.step_ramp_font_offset_x * scale_factor
        self.step_ramp_font_offset_y\
            = param.step_ramp_font_offset_y * scale_factor
        self.sound_text_font_size\
            = param.sound_text_font_size * scale_factor
        self.sound_text_font_path = param.sound_text_font_path
        self.dummy_img_size = 1024 * scale_factor

    def _debug_dump_param(self):
        for key, value in self.__dict__.items():
            print(key, ':', value)

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
        self.ramp_pos_h\
            = (self.width // 2) - (ramp_width // 2) - self.ramp_outline_width
        tpg.merge(
            self.img, ramp_obj_img, pos=(self.ramp_pos_h,
                                         self.step_ramp_pos_v))

    def draw_text_into_step_ramp(self, block_img, code_value):
        fg_color = (1 - code_value) / 2
        text = "{:>4d}".format(int(code_value * self.code_value_max + 0.5))
        text_drawer = TextDrawer(
            block_img, text,
            pos=(self.step_ramp_font_offset_x, self.step_ramp_font_offset_y),
            font_color=(fg_color, fg_color, fg_color),
            font_size=self.step_ramp_font_size,
            transfer_functions=self.transfer_function,
            font_path=NOTO_SANS_MONO_BOLD)
        text_drawer.draw()

    def get_text_size(
            self, text="0", font_size=10, font_path=NOTO_SANS_MONO_BOLD):
        """
        テキスト1文字分の width, height を求める。

        example
        =======
        >>> width, height = self.get_text_size(
        >>>     text="0", font_size=10, font_path=NOTO_SANS_MONO_BOLD)
        """
        dummy_img = np.zeros((self.dummy_img_size, self.dummy_img_size, 3))
        text_drawer = TextDrawer(
            dummy_img, text=text, pos=(0, 0),
            font_color=self.fg_color/0xFF,
            font_size=font_size,
            transfer_functions=self.transfer_function,
            font_path=font_path)
        text_drawer.draw()
        return text_drawer.get_text_size()

    def draw_sound_text(self, text="L"):
        width, height = self.get_text_size(
            text="0", font_size=self.sound_text_font_size,
            font_path=self.sound_text_font_path)

        upper_left\
            = (self.ramp_pos_h + width // 4,
               self.step_ramp_pos_v + self.ramp_obj_height + height // 4)
        upper_right\
            = (self.width - (self.ramp_pos_h + width // 4) - width,
               self.step_ramp_pos_v + self.ramp_obj_height + height // 4)
        lower_left\
            = (self.ramp_pos_h + width // 4,
               self.height - (self.step_ramp_pos_v + self.ramp_obj_height + height // 4) - height)
        lower_right\
            = (self.width - (self.ramp_pos_h + width // 4) - width,
               self.height - (self.step_ramp_pos_v + self.ramp_obj_height + height // 4) - height)

        text_drawer = TextDrawer(
            self.img, text, pos=upper_left,
            font_color=self.sound_text_color,
            font_size=self.sound_text_font_size,
            transfer_functions=self.transfer_function,
            font_path=self.sound_text_font_path)
        text_drawer.draw()

        text_drawer = TextDrawer(
            self.img, text, pos=upper_right,
            font_color=self.sound_text_color,
            font_size=self.sound_text_font_size,
            transfer_functions=self.transfer_function,
            font_path=self.sound_text_font_path)
        text_drawer.draw()

        text_drawer = TextDrawer(
            self.img, text, pos=lower_left,
            font_color=self.sound_text_color,
            font_size=self.sound_text_font_size,
            transfer_functions=self.transfer_function,
            font_path=self.sound_text_font_path)
        text_drawer.draw()

        text_drawer = TextDrawer(
            self.img, text, pos=lower_right,
            font_color=self.sound_text_color,
            font_size=self.sound_text_font_size,
            transfer_functions=self.transfer_function,
            font_path=self.sound_text_font_path)
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
        self.draw_sound_text(text="C")

        # tpg.preview_image(self.img)

    def save(self):
        cv2.imwrite(
            self.filename, np.uint16(np.round(self.img[:, :, ::-1] * 0xFFFF)))


class CountDownImageColorParam(NamedTuple):
    transfer_function: str = tf.GAMMA24
    bg_luminance: float = 18.0
    fg_luminance: float = 90.0
    object_outline_luminance: float = 1.0


class CountDownImageCoordinateParam(NamedTuple):
    radius1: int = 300
    radius2: int = 295
    radius3: int = 290
    radius4: int = 280
    fps: int = 24
    crosscross_line_width: int = 4
    font_size: int = 60
    font_path: str = NOTO_SANS_MONO_BOLD


class CountDownSequence():
    def __init__(
            self, color_param, coordinate_param, fname_base, dynamic_range,
            scale_factor):
        self.transfer_function = color_param.transfer_function
        self.fg_color = self.convert_luminance_to_color_value(
            color_param.fg_luminance, self.transfer_function)
        self.bg_color = self.convert_luminance_to_color_value(
            color_param.bg_luminance, self.transfer_function)
        self.obj_outline_color = self.convert_luminance_to_color_value(
            color_param.object_outline_luminance, self.transfer_function)

        self.set_coordinate_param(coordinate_param, scale_factor)

        self.fname_base = fname_base
        self.fname_width = 1920 * scale_factor
        self.fname_height = 1080 * scale_factor
        self.dynamic_range = dynamic_range

        # self._debug_dump_param()

        self.counter = 0

    def _debug_dump_param(self):
        for key, value in self.__dict__.items():
            print(key, ':', value)

    def convert_luminance_to_color_value(self, x, y):
        return self.float_to_uint8(
            tpg.convert_luminance_to_color_value(x, y))

    def float_to_uint8(self, x):
        return np.uint8(np.round(x * 0xFF))

    def set_coordinate_param(self, param, scale_factor):
        self.fps = param.fps
        self.space = 4 * scale_factor
        self.radius1 = param.radius1 * scale_factor
        self.radius2 = param.radius2 * scale_factor
        self.radius3 = param.radius3 * scale_factor
        self.radius4 = param.radius4 * scale_factor
        self.cc_line_width = param.crosscross_line_width * scale_factor
        self.img_width = self.radius1 * 2 + self.space * 2
        self.img_height = self.img_width
        self.center_pos = (self.img_width // 2, self.img_height // 2)
        self.font_size = param.font_size * scale_factor
        self.font_path = param.font_path

    def calc_text_pos(self, text="0"):
        dummy_img = np.zeros((self.img_height, self.img_width, 3))
        text_drawer = TextDrawer(
            dummy_img, text=text, pos=(0, 0),
            font_color=self.fg_color/0xFF,
            font_size=self.font_size,
            transfer_functions=self.transfer_function,
            font_path=self.font_path)
        text_drawer.draw()
        text_width, text_height = text_drawer.get_text_size()
        pos_h = self.img_width // 2 - text_width // 2
        pos_v = self.img_height // 2 - text_height // 2
        self.font_pos = (pos_h, pos_v)

    def draw_crisscross_line(self, img):
        # H Line
        height, width = img.shape[0:2]
        pos_v = self.center_pos[1] - self.cc_line_width // 2
        pt1 = (0, pos_v)
        pt2 = (width, pos_v)
        tpg.draw_straight_line(
            img, pt1, pt2, self.obj_outline_color, self.cc_line_width)

        # V Line
        pos_h = self.center_pos[0] - self.cc_line_width // 2
        pt1 = (pos_h, 0)
        pt2 = (pos_h, height)
        tpg.draw_straight_line(
            img, pt1, pt2, self.obj_outline_color, self.cc_line_width)

    def draw_circles(self, img):
        cv2.circle(
            img, self.center_pos, self.radius1, self.fg_color.tolist(), -1,
            cv2.LINE_AA)
        cv2.circle(
            img, self.center_pos, self.radius2,
            self.obj_outline_color.tolist(), -1, cv2.LINE_AA)
        cv2.circle(
            img, self.center_pos, self.radius3,
            self.bg_color.tolist(), -1, cv2.LINE_AA)

    def draw_ellipse(self, img, frame=10):
        end_angle = 360 / self.fps * frame - 90
        cv2.ellipse(
            img, self.center_pos, (self.radius4, self.radius4), angle=0,
            startAngle=-90, endAngle=end_angle,
            color=self.obj_outline_color.tolist(), thickness=-1)

    def draw_text(self, img, sec):
        self.calc_text_pos(text=str(sec))
        text_drawer = TextDrawer(
            img / 0xFF, text=str(sec), pos=self.font_pos,
            font_color=self.fg_color / 0xFF,
            font_size=self.font_size,
            transfer_functions=self.transfer_function,
            font_path=self.font_path)
        text_drawer.draw()
        return text_drawer.get_img()

    def attatch_alpha_channel(self, img):
        dummy = np.zeros_like(img, dtype=np.uint8)
        cv2.circle(
            dummy, self.center_pos, self.radius1, [0xFF, 0xFF, 0xFF], -1,
            cv2.LINE_AA)
        alpha = dummy[..., 1] / 0xFF
        img = np.dstack((img, alpha))
        return img

    def draw_countdown_seuqence_image(self, sec, frame):
        img = np.ones((self.img_height, self.img_width, 3), dtype=np.uint8)
        self.draw_circles(img)
        self.draw_crisscross_line(img)
        self.draw_ellipse(img, frame=frame)

        self.filename = self.fname_base.format(
            self.dynamic_range, self.fname_width, self.fname_height,
            self.counter)

        # ここから img は float
        img = self.draw_text(img, sec)
        img = self.attatch_alpha_channel(img)

        self.counter += 1

        # cv2.imwrite(self.filename, np.uint16(np.round(img * 0xFFFF)))

        return img
