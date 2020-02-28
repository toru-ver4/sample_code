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


CHROMA = {'ITU-R BT.709': [0.467893566324554, 0.215787584674837,
                           0.149781375011629, 0.208001181214428,
                           0.139928129463102, 0.137355386982480,
                           0.317511492648874, 0.754346115001794]}

CHROMA_MAX_02_00 = 2.74710773964960
CHROMA_MAX_03_00 = 4.12066160947440
CHROMA_MAX_04_00 = 5.49421547929920
CHROMA_MAX_05_00 = 6.86776934912400
CHROMA_MAX_07_00 = 9.35495271612193
CHROMA_MAX_10_00 = 11.7856223341119
CHROMA_MAX_58_00 = 33.5436943355494

# # L* = 0.5
CHROMA = {
    'ITU-R BT.709': [2.33946783162278, 1.07893792337419, 0.748906875058145,
                     1.04000590607214, 0.699640647315515, 0.686776934912400,
                     1.58755746324438, 3.77173057500899]}
# L* = 1.0
CHROMA = {
    'ITU-R BT.709': [4.67893566324554, 2.15787584674837, 1.49781375011629,
                     2.08001181214428, 1.39928129463102, 1.37355386982480,
                     3.17511492648874, 7.54346115001794]}
# L* = 2.0
CHROMA = {
    'ITU-R BT.709': [9.35787132649113, 4.31575169349677, 2.99562750023258,
                     4.16002362428858, 2.79856258926206, 2.74710773964960,
                     6.35022985297751, 15.0874246637395]}
# L* = 3.0
CHROMA = {
    'ITU-R BT.709': [14.0368069897367, 6.47362754024514, 4.49344125034887,
                     6.24003543643286, 4.19784388389309, 4.12066160947440,
                     9.50625735073427, 22.9227504958066]}
# L* = 4.0
CHROMA = {
    'ITU-R BT.709': [18.6952150995068, 8.63150338699351, 5.99125500046515,
                     8.32004724857714, 5.59712517852411, 5.49421547929920,
                     12.0357588702337, 31.0388478042729]}
# # L* = 5.0
CHROMA = {
    'ITU-R BT.709': [22.4851328201947, 10.7893792337419, 7.48906875058145,
                     10.4000590607214, 6.99640647315515, 6.86776934912400,
                     13.9036573539259, 37.2973759306897]}


# # L* = 10
# CHROMA = {
#     'ITU-R BT.709': [31.1134446845507, 21.6612897565317, 14.9393690072093,
#                      20.7274414737387, 13.4941585448319, 11.7856223341119,
#                      18.6662114728803, 51.6143768309843]}


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

        if text == "R" or text == "C":
            text_drawer = TextDrawer(
                self.img, text, pos=upper_right,
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
            transfer_functions=self.transfer_function,
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
            transfer_functions=self.transfer_function,
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
        width = text_width + padding * 2
        height = text_height + padding * 2
        img = np.zeros((height, width, 3))

        # Low Level(64) の描画＆合成
        text_drawer = TextDrawer(
            img, low_text, pos=(padding, padding),
            font_color=low_text_color,
            font_size=self.limited_text_font_size,
            transfer_functions=self.transfer_function,
            font_path=self.limited_text_font_path)
        text_drawer.draw()
        st_pos_h = self.ramp_pos_h // 2 - width // 2
        st_pos_v = self.step_ramp_pos_v + self.ramp_obj_height // 2\
            - height // 2
        tpg.merge(self.img, img, pos=(st_pos_h, st_pos_v))

        # High Level(940) の描画＆合成
        img = np.ones_like(img)
        text_drawer = TextDrawer(
            img, high_text, pos=(padding, padding),
            font_color=high_text_color,
            font_size=self.limited_text_font_size,
            transfer_functions=self.transfer_function,
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
        # テキストを収める箱を作る
        text_width, text_height = self.get_text_size(
            text="R", font_size=self.dot_dropped_text_size,
            font_path=self.limited_text_font_path)
        padding = text_height // 6
        width = text_width * 2 + padding * 3
        height = text_height * 2 + padding * 3
        img = np.ones((height, width, 3)) * self.obj_outline_color

        texts = ["W", "R", "G", "M"]
        text_colors = np.array(
            [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])

        # テキスト描画
        for idx in range(len(texts)):
            text = texts[idx]
            text_color = text_colors[idx]
            pos_h = padding + (padding + text_width) * (idx % 2)
            pos_v = padding + (padding + text_height) * (idx // 2)
            text_drawer = TextDrawer(
                img, text=text, pos=(pos_h, pos_v),
                font_color=text_color,
                font_size=self.dot_dropped_text_size,
                transfer_functions=self.transfer_function,
                font_path=self.limited_text_font_path)
            text_drawer.draw_with_dropped_dot(dot_factor=2)

        # 背景画像と合成
        pos_h = self.limited_range_high_center_pos_h - width // 2
        temp = ((self.height // 2) - self.limited_range_ed_pos_v) // 2
        pos_v = self.limited_range_ed_pos_v + temp - height // 2
        tpg.merge(self.img, img, ((pos_h & 0xFFFE) + 1, (pos_v & 0xFFFE) + 1))

    def draw_low_level_color_patch(self):
        outmost_num = 5
        total_width = self.lab_patch_each_size * outmost_num
        total_height = total_width
        img = np.zeros((total_height, total_width, 3))
        rgb = tpg.calc_same_lstar_radial_color_patch_data(
            lstar=7.0, chroma=CHROMA_MAX_07_00, outmost_num=outmost_num,
            color_space=RGB_COLOURSPACES[self.gamut],
            transfer_function=self.transfer_function)

        for idx in range(outmost_num ** 2):
            h_idx = idx % outmost_num
            v_idx = idx // outmost_num
            st_pos = (h_idx * self.lab_patch_each_size,
                      v_idx * self.lab_patch_each_size)
            temp_img = np.ones((self.lab_patch_each_size,
                                self.lab_patch_each_size, 3))\
                * rgb[idx][np.newaxis, np.newaxis, :]
            tpg.merge(img, temp_img, st_pos)

        pos_h = self.ramp_pos_h // 2 - total_width // 2
        temp = ((self.height // 2) - self.limited_range_ed_pos_v) // 2
        pos_v = self.limited_range_ed_pos_v + temp - total_height // 2
        pos = (pos_h, pos_v)
        tpg.merge(self.img, img, pos)

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
        self.draw_low_level_color_patch()
        # self.draw_low_level_color_patch(l_val=10)

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
