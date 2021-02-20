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
from sympy import symbols
from colour import RGB_COLOURSPACES

# import my libraries
import transfer_functions as tf
import test_pattern_generator2 as tpg
from font_control import TextDrawer
from font_control import NOTO_SANS_MONO_BOLD, NOTO_SANS_MONO_BLACK,\
    NOTO_SANS_MONO_REGULAR

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
    gamut: str = 'ITU-R BT.709'
    text_info_luminance: float = 50.0
    crosshatch_luminance: float = 5.0
    checker_board_levels: list = [
        [504, 506], [506, 508], [508, 510], [510, 512]]
    ramp_10bit_levels: list = [400, 425]
    dot_droped_luminance: float = 90.0


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
    info_text_font_size: float = 10
    info_text_font_path: str = NOTO_SANS_MONO_REGULAR
    limited_text_font_size: float = 100
    limited_text_font_path: str = NOTO_SANS_MONO_BLACK
    crosshatch_size: int = 128
    dot_dropped_text_size: float = 100
    lab_patch_each_size: int = 32
    even_odd_info_text_size: int = 10
    ramp_10bit_info_text_size: int = 10
    tp_obj_outline_width: int = 1


def convert_from_pillow_to_numpy(img):
    img = np.asarray(img) / 0xFF

    return img


class BackgroundImage():
    def __init__(
            self, color_param, coordinate_param, fname_base, dynamic_range,
            scale_factor, fps, revision):
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
        self.gamut = color_param.gamut
        self.text_info_color = tpg.convert_luminance_to_color_value(
            color_param.text_info_luminance, self.transfer_function)
        self.crosshatch_color = tpg.convert_luminance_to_color_value(
            color_param.crosshatch_luminance, self.transfer_function)
        self.checker_board_levels = color_param.checker_board_levels
        self.ramp_10bit_levels = color_param.ramp_10bit_levels
        self.dot_droped_code_value = tpg.convert_luminance_to_code_value(
            color_param.dot_droped_luminance, self.transfer_function)

        # text settings
        self.__sound_text = " "
        self.__frame_idx = 0
        self.fps = fps
        self.dynamic_range = dynamic_range
        self.revision = revision

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
        self.into_text_font_size\
            = param.info_text_font_size * scale_factor
        self.info_text_font_path = param.info_text_font_path
        self.limited_text_font_size\
            = param.limited_text_font_size * scale_factor
        self.limited_text_font_path = param.limited_text_font_path
        self.crosshatch_size = param.crosshatch_size * scale_factor
        self.dot_dropped_text_size\
            = param.dot_dropped_text_size * scale_factor
        self.lab_patch_each_size\
            = param.lab_patch_each_size * scale_factor
        self.even_odd_info_text_size\
            = param.even_odd_info_text_size * scale_factor
        self.ramp_10bit_info_text_size\
            = param.ramp_10bit_info_text_size * scale_factor
        self.tp_obj_outline_width\
            = param.tp_obj_outline_width * scale_factor

    @property
    def sound_text(self):
        return self.__sound_text

    @sound_text.setter
    def sound_text(self, text):
        self.__sound_text = text

    @property
    def frame_idx(self):
        return self.__frame_idx

    @frame_idx.setter
    def frame_idx(self, frame_idx):
        self.__frame_idx = frame_idx

    @property
    def is_even_number(self):
        return self.__is_even_number

    @is_even_number.setter
    def is_even_number(self, is_even_number):
        self.__is_even_number = is_even_number

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
            bg_transfer_functions=self.transfer_function,
            fg_transfer_functions=tf.GAMMA24,
            font_path=NOTO_SANS_MONO_BLACK)
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
            bg_transfer_functions=self.transfer_function,
            font_path=font_path)
        text_drawer.draw()
        return text_drawer.get_text_size()

    def draw_sound_text(self, text=" "):
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
               self.height - (self.step_ramp_pos_v + self.ramp_obj_height
                              + height // 4) - height)
        lower_right\
            = (self.width - (self.ramp_pos_h + width // 4) - width,
               self.height - (self.step_ramp_pos_v + self.ramp_obj_height
                              + height // 4) - height)

        if text == "L" or text == "C":
            text_drawer = TextDrawer(
                self.img, text, pos=upper_left,
                font_color=self.sound_text_color,
                font_size=self.sound_text_font_size,
                fg_transfer_functions=self.transfer_function,
                bg_transfer_functions=self.transfer_function,
                font_path=self.sound_text_font_path)
            text_drawer.draw()

            text_drawer = TextDrawer(
                self.img, text, pos=lower_left,
                font_color=self.sound_text_color,
                font_size=self.sound_text_font_size,
                fg_transfer_functions=self.transfer_function,
                bg_transfer_functions=self.transfer_function,
                font_path=self.sound_text_font_path)
            text_drawer.draw()

        if text == "R" or text == "C":
            text_drawer = TextDrawer(
                self.img, text, pos=upper_right,
                font_color=self.sound_text_color,
                font_size=self.sound_text_font_size,
                fg_transfer_functions=self.transfer_function,
                bg_transfer_functions=self.transfer_function,
                font_path=self.sound_text_font_path)
            text_drawer.draw()

            text_drawer = TextDrawer(
                self.img, text, pos=lower_right,
                font_color=self.sound_text_color,
                font_size=self.sound_text_font_size,
                fg_transfer_functions=self.transfer_function,
                bg_transfer_functions=self.transfer_function,
                font_path=self.sound_text_font_path)
            text_drawer.draw()

    def draw_signal_information(self):
        """
        解像度とか色域とかの情報
        """
        text_base = "{}x{}, {:02d}fps, {}, {}, D65"
        text = text_base.format(
            self.width, self.height, self.fps, self.transfer_function,
            self.gamut)
        width, height = self.get_text_size(
            text=text, font_size=self.into_text_font_size,
            font_path=self.info_text_font_path)

        st_pos_v = self.height - self.outline_width * 4 - height
        st_pos_h = self.outline_width * 4

        text_drawer = TextDrawer(
            self.img, text, pos=(st_pos_h, st_pos_v),
            font_color=self.text_info_color,
            font_size=self.into_text_font_size,
            fg_transfer_functions=self.transfer_function,
            bg_transfer_functions=self.transfer_function,
            font_path=self.info_text_font_path)
        text_drawer.draw()

    def draw_revision(self):
        """
        解像度とか色域とかの情報
        """
        text = "Revision {:02d}".format(self.revision)
        width, height = self.get_text_size(
            text=text, font_size=self.into_text_font_size,
            font_path=self.info_text_font_path)

        st_pos_v = self.height - self.outline_width * 4 - height
        st_pos_h = self.width - self.outline_width * 4 - width

        text_drawer = TextDrawer(
            self.img, text, pos=(st_pos_h, st_pos_v),
            font_color=self.text_info_color,
            font_size=self.into_text_font_size,
            fg_transfer_functions=self.transfer_function,
            bg_transfer_functions=self.transfer_function,
            font_path=self.info_text_font_path)
        text_drawer.draw()

    def draw_information(self):
        """
        画面下部にテキストを書く
        """
        self.draw_signal_information()
        self.draw_revision()

    def draw_limited_range_text(self):
        """
        Limited Range の OK/NG 判別用のテキスト描画
        """
        low_text_color = np.array([64, 64, 64]) / 1023
        high_text_color = np.array([940, 940, 940]) / 1023
        low_text = "{:03d}".format(64)
        high_text = "{:03d}".format(940)

        # テキストを収める箱を作る
        text_width, text_height = self.get_text_size(
            text=low_text, font_size=self.limited_text_font_size,
            font_path=self.limited_text_font_path)
        padding = text_height // 4
        width = int(text_width + padding * 2.0)
        height = text_height + padding * 2
        img = np.zeros((height, width, 3))

        # Low Level(64) の描画＆合成
        text_drawer = TextDrawer(
            img, low_text, pos=(padding, padding),
            font_color=low_text_color,
            font_size=self.limited_text_font_size,
            fg_transfer_functions=self.transfer_function,
            bg_transfer_functions=self.transfer_function,
            font_path=self.limited_text_font_path)
        text_drawer.draw()
        st_pos_h = self.ramp_pos_h // 2 - width // 2
        st_pos_v = self.step_ramp_pos_v + self.ramp_obj_height // 2\
            - height // 2
        tpg.merge(self.img, img, pos=(st_pos_h, st_pos_v))
        self.limited_range_low_center_pos_h = st_pos_h + width // 2

        # High Level(940) の描画＆合成
        img = np.ones_like(img)
        text_drawer = TextDrawer(
            img, high_text, pos=(padding, padding),
            font_color=high_text_color,
            font_size=self.limited_text_font_size,
            fg_transfer_functions=self.transfer_function,
            bg_transfer_functions=self.transfer_function,
            font_path=self.limited_text_font_path)
        text_drawer.draw()
        st_pos_h = self.ramp_pos_h // 2 - width // 2
        st_pos_v = self.step_ramp_pos_v + self.ramp_obj_height // 2\
            - height // 2
        tpg.merge(self.img, img,
                  pos=(self.width - st_pos_h - width, st_pos_v))

        self.limited_range_ed_pos_v = st_pos_v + height
        self.limited_range_high_center_pos_h\
            = self.width - st_pos_h - width // 2

    def draw_crosshatch(self):
        st_pos_v = self.height // 2
        st_pos_h = self.width // 2

        # v-direction loop
        v_loop_num = int((self.height / 2) / self.crosshatch_size + 0.5) + 1
        for v_idx in range(1, v_loop_num):
            pos_v_upper = st_pos_v - v_idx * self.crosshatch_size
            pos_v_lower = st_pos_v + v_idx * self.crosshatch_size
            tpg.draw_straight_line(
                self.img, (0, pos_v_upper), (self.width, pos_v_upper),
                self.crosshatch_color, 1)
            tpg.draw_straight_line(
                self.img, (0, pos_v_lower), (self.width, pos_v_lower),
                self.crosshatch_color, 1)

        # h-direction loop
        h_loop_num = int((self.width / 2) / self.crosshatch_size + 0.5)
        for h_idx in range(1, h_loop_num):
            pos_h_upper = st_pos_h - h_idx * self.crosshatch_size
            pos_h_lower = st_pos_h + h_idx * self.crosshatch_size
            tpg.draw_straight_line(
                self.img, (pos_h_upper, 0), (pos_h_upper, self.height),
                self.crosshatch_color, 1)
            tpg.draw_straight_line(
                self.img, (pos_h_lower, 0), (pos_h_lower, self.height),
                self.crosshatch_color, 1)

    def draw_dot_dropped_text(self):
        """
        Make the Dot-Dropped pattern.
        This is used to distinguish if it is 4:4:4 or 4:2:2 or 4:2:0.
        """
        # define
        dot_factor = 3  # 4x4 pixel
        drop_pixel = (2 ** dot_factor) // 2
        pos_mask = 0x10000 - drop_pixel   # 4x4 pixel
        dot_offset = (-1, -1)

        texts = ["W", "Y", "C", "G", "M", "R", "B"]
        pos_info_even = "Start with\neven numbers"
        pos_info_odd = "Start with\nodd numbers"
        text_colors = np.array(
            [[1., 1., 1.], [1., 1., 0.], [0., 1., 1.],
             [0., 1., 0.], [1., 0., 1.], [1., 0., 0.], [0, 0, 1.]])\
            * self.dot_droped_code_value

        text_width, text_height = self.get_text_size(
            text="R", font_size=self.dot_dropped_text_size,
            font_path=self.limited_text_font_path)
        padding_factor = 6
        padding = ((text_height // padding_factor) // drop_pixel) * drop_pixel
        width = text_width + padding * 2
        self.dot_drop_width = width
        height = text_height * len(texts) + padding * (len(texts) + 1)
        img_even = np.ones((height, width, 3)) * self.bg_color / 2
        img_odd = np.ones_like(img_even) * self.bg_color / 2

        # テキスト描画
        for idx in range(len(texts)):
            text = texts[idx]
            text_color = text_colors[idx]
            pos_h = padding
            pos_v = pos_mask & (padding + (padding + text_height) * idx)
            text_drawer_even = TextDrawer(
                img_even, text=text, pos=(pos_h, pos_v),
                font_color=text_color,
                font_size=self.dot_dropped_text_size,
                bg_transfer_functions=self.transfer_function,
                fg_transfer_functions=self.transfer_function,
                font_path=self.limited_text_font_path)
            text_drawer_even.draw_with_dropped_dot(dot_factor=dot_factor)

            text_drawer_odd = TextDrawer(
                img_odd, text=text, pos=(pos_h, pos_v),
                font_color=text_color,
                font_size=self.dot_dropped_text_size,
                bg_transfer_functions=self.transfer_function,
                fg_transfer_functions=self.transfer_function,
                font_path=self.limited_text_font_path)
            text_drawer_odd.draw_with_dropped_dot(
                dot_factor=dot_factor, offset=dot_offset)

        # 背景画像と合成
        temp = ((self.height // 2) - self.limited_range_ed_pos_v) // 2
        pos_v = pos_mask & (self.limited_range_ed_pos_v + temp // 2)
        pos_h_left_img =\
            pos_mask &\
            (self.limited_range_high_center_pos_h - drop_pixel // 2 - width)
        pos_h_right_img =\
            pos_mask &\
            (self.limited_range_high_center_pos_h + drop_pixel // 2)

        if not self.is_even_number:
            left_img = img_even
            right_img = img_odd
            left_info = pos_info_even
            right_info = pos_info_odd
        else:
            left_img = img_odd
            right_img = img_even
            left_info = pos_info_odd
            right_info = pos_info_even

        tpg.merge(self.img, left_img, (pos_h_left_img, pos_v))
        tpg.merge(self.img, right_img, (pos_h_right_img, pos_v))
        self.dot_drop_ed_pos_v = pos_v + right_img.shape[0]

        # EVEN, ODD の情報をテキストとして記述
        text_width, text_height = self.get_text_size(
            text="R", font_size=self.even_odd_info_text_size,
            font_path=self.info_text_font_path)
        info_pos_v = int(pos_v - text_height * 3.7)
        info_left_pos_h = pos_h_left_img
        info_fight_pos_h = pos_h_right_img

        # left
        text_draw_left = TextDrawer(
            self.img, text=left_info, pos=(info_left_pos_h, info_pos_v),
            font_color=self.text_info_color,
            font_size=self.even_odd_info_text_size,
            bg_transfer_functions=self.transfer_function,
            fg_transfer_functions=self.transfer_function,
            font_path=self.info_text_font_path
        )
        text_draw_left.draw()

        # right
        text_draw_left = TextDrawer(
            self.img, text=right_info, pos=(info_fight_pos_h, info_pos_v),
            font_color=self.text_info_color,
            font_size=self.even_odd_info_text_size,
            bg_transfer_functions=self.transfer_function,
            fg_transfer_functions=self.transfer_function,
            font_path=self.info_text_font_path
        )
        text_draw_left.draw()

    def draw_10bit_detection(self):
        """
        Draw the checker board to distinguish
        if it is displayed at 10 bit depth.
        """
        text_width, text_height = self.get_text_size(
            text="10bit", font_size=self.ramp_10bit_info_text_size,
            font_path=self.info_text_font_path)
        text_v_margin = text_height * 3
        block_num = len(self.checker_board_levels)
        temp = ((self.height // 2) - self.limited_range_ed_pos_v) // 2
        st_pos_v = self.limited_range_ed_pos_v + temp // 2
        total_v = self.dot_drop_ed_pos_v - st_pos_v
        block_height = (total_v - text_v_margin * (block_num - 1)) // block_num
        block_width = block_height
        st_pos_h = self.limited_range_low_center_pos_h - block_width // 2

        for block_idx in range(block_num):
            # draw checker board
            low_level = [self.checker_board_levels[block_idx][0]
                         for x in range(3)]
            high_level = [self.checker_board_levels[block_idx][1]
                          for x in range(3)]
            checker_board_img = tpg.make_tile_pattern(
                width=block_width, height=block_height,
                h_tile_num=4, v_tile_num=4,
                low_level=high_level, high_level=low_level) / 1023
            tpg.draw_outline(checker_board_img, self.obj_outline_color, 1)

            pos_v = st_pos_v + block_idx * (block_height + text_v_margin)
            tpg.merge(self.img, checker_board_img, (st_pos_h, pos_v))

            # add text
            text_pos_v = pos_v - int(text_height * 1.5)
            text_draw_left = TextDrawer(
                self.img, text=f"{low_level[0]} Lv, {high_level[0]} Lv",
                pos=(st_pos_h, text_pos_v),
                font_color=self.text_info_color,
                font_size=self.ramp_10bit_info_text_size,
                bg_transfer_functions=self.transfer_function,
                fg_transfer_functions=self.transfer_function,
                font_path=self.info_text_font_path
            )
            text_draw_left.draw()

    def draw_10bit_v_ramp(self):
        """
        Draw the ramp pattern to distinguish
        if it is displayed at 10 bit depth.
        """
        if not self.is_even_number:
            bit_depth_list = [6, 8, 10]
        else:
            bit_depth_list = [6, 10, 8]
        mask_list = [0x10000 - (2 ** (10 - x)) for x in bit_depth_list]

        temp = ((self.height // 2) - self.limited_range_ed_pos_v) // 2
        st_pos_v = self.limited_range_ed_pos_v + temp // 2
        total_width = int(self.dot_drop_width * 3)
        width = total_width\
            - self.tp_obj_outline_width * 2 * (len(bit_depth_list) - 1)
        width = width // len(bit_depth_list)
        width_with_margin = width + self.tp_obj_outline_width * 2
        height = self.dot_drop_ed_pos_v - st_pos_v
        pos_st_h_base = self.limited_range_low_center_pos_h - total_width // 2

        grad = np.linspace(
            self.ramp_10bit_levels[0], self.ramp_10bit_levels[1], height)
        grad = np.dstack((grad, grad, grad)).reshape((height, 1, 3))
        ramp_base = np.ones((height, width, 3)) * grad

        text_width, text_height = self.get_text_size(
            text="10bit", font_size=self.ramp_10bit_info_text_size,
            font_path=self.info_text_font_path)
        info_pos_v = int(st_pos_v - text_height * 1.3)

        for idx in range(len(mask_list)):
            ramp_xx_bit = (np.uint16(ramp_base) & mask_list[idx]) / 1023
            tpg.draw_outline(
                ramp_xx_bit, self.obj_outline_color, self.tp_obj_outline_width)
            pos_st_h = pos_st_h_base + width_with_margin * idx
            tpg.merge(self.img, ramp_xx_bit, (pos_st_h, st_pos_v))

            # 8bit, 10bit の情報をテキストとして記述

            text_draw_left = TextDrawer(
                self.img, text=f"{bit_depth_list[idx]}bit",
                pos=(pos_st_h, info_pos_v),
                font_color=self.text_info_color,
                font_size=self.ramp_10bit_info_text_size,
                bg_transfer_functions=self.transfer_function,
                fg_transfer_functions=self.transfer_function,
                font_path=self.info_text_font_path
            )
            text_draw_left.draw()

    def draw_8bit_10bit_identification_patterns(self):
        """
        8bit と 10bit の識別パターンを描画する
        """
        even_mask = 0x100000000 - 2
        total_width = int(self.dot_drop_width * 3) & even_mask
        patch_height = (total_width // 3) % even_mask
        patch_internal_margin_v = 2
        patch_rest_margin_v = (patch_height // 2) & even_mask
        patch_pos_v_offset\
            = patch_height * 2 + patch_rest_margin_v + patch_internal_margin_v
        temp = ((self.height // 2) - self.limited_range_ed_pos_v) // 2
        st_pos_v = (self.limited_range_ed_pos_v + temp // 2) % even_mask
        pos_st_h_base = self.limited_range_low_center_pos_h - total_width // 2
        text_width, text_height = self.get_text_size(
            text="10bit", font_size=self.ramp_10bit_info_text_size,
            font_path=self.info_text_font_path)

        patch_width = int(total_width - text_width * 1.2) % even_mask
        patch_pos_h = int(pos_st_h_base + text_width * 1.2) % even_mask
        patch_pos_v = st_pos_v
        info_pos_h = pos_st_h_base
        info_pos_v = patch_pos_v

        dummy_img = np.ones((patch_height, patch_width, 3))
        bit_depth_list = [8, 10]

        for level in ['low', 'middle', 'high']:
            for idx in range(2):
                text_draw_left = TextDrawer(
                    self.img, text=f"{bit_depth_list[idx]}bit",
                    pos=(
                        info_pos_h,
                        info_pos_v + patch_internal_margin_v * idx
                        + patch_height * idx),
                    font_color=self.text_info_color,
                    font_size=self.ramp_10bit_info_text_size,
                    bg_transfer_functions=self.transfer_function,
                    fg_transfer_functions=self.transfer_function,
                    font_path=self.info_text_font_path
                )
                text_draw_left.draw()
                tpg.merge(
                    self.img, dummy_img * idx,
                    (patch_pos_h,
                     patch_pos_v + patch_internal_margin_v * idx
                     + patch_height * idx))
            patch_pos_v += patch_pos_v_offset
            info_pos_v = patch_pos_v

    def make(self):
        """
        背景画像を生成する
        """
        self.img = np.ones((self.height, self.width, 3))
        self.img = self.img * self.bg_color
        self.draw_crosshatch()
        self.draw_crisscross_line()
        self.draw_outline(self.img, self.fg_color, self.outline_width)
        self.draw_ramp_pattern()
        self.draw_step_ramp_pattern()
        self.draw_sound_text(self.sound_text)
        self.draw_information()
        self.draw_limited_range_text()
        self.draw_dot_dropped_text()
        # self.draw_10bit_detection()
        self.draw_10bit_v_ramp()
        self.draw_8bit_10bit_identification_patterns()

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
            fg_transfer_functions=self.transfer_function,
            bg_transfer_functions=self.transfer_function,
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
            bg_transfer_functions=self.transfer_function,
            fg_transfer_functions=self.transfer_function,
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
