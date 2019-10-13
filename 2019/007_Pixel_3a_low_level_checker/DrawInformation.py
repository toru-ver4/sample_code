#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
情報を記入
"""

# 外部ライブラリのインポート
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# 自作ライブラリのインポート
import test_pattern_generator2 as tpg
import transfer_functions as tf

# define
font_luminance = 70  # unit is [cd/m2].
font_level = int(tf.oetf_from_luminance(font_luminance, tf.SRGB) * 0xFF + 0.5)
font_size_rate = 0.01
st_pos_offset = 0.007
font_path = "/usr/share/fonts/opentype/noto/NotoSansMonoCJKjp-Regular.otf"
png_file_name = './intermediate/diagram.png'


class DrawInformation:
    """
    特記事項なし
    """
    def __init__(self, base_param, draw_param, img,
                 diagram_width, diagram_height):
        self.base_param = base_param
        self.draw_param = draw_param
        self.img = img
        self.diagram_width = diagram_width
        self.diagram_height = diagram_height

    def get_img_width(self):
        return self.img.shape[1]

    def get_img_height(self):
        return self.img.shape[0]

    def int(self, x):
        return int(x + 0.5)

    def make_info_text(self):
        outer_gamut_name = self.base_param['outer_gamut_name']
        inner_gamut_name = self.base_param['inner_gamut_name']
        transfer_function_name = self.base_param['transfer_function']
        reference_white_name = self.base_param['reference_white']
        revision = "■Revision\n  {:02}".format(self.base_param['revision'])
        outer_gamut = self.make_primariey_info_text(
            '■Outer', outer_gamut_name, self.base_param['outer_primaries'])
        inner_gamut = self.make_primariey_info_text(
            '■Inner', inner_gamut_name, self.base_param['inner_primaries'])
        transfer_function =\
            "■Transfer Function\n  {}".format(transfer_function_name)
        reference_white =\
            "■Reference White:\n  {} nits".format(reference_white_name)

        return "\n".join([revision, outer_gamut, inner_gamut,
                          transfer_function, reference_white])

    # def make_primariey_info_text(self, prefix, name, primaries):
    #     t = "{}Gamut\n  name: {}\n  xy  : [{}, {}]\n{}[{}, {}]\n{}[{}, {}]"
    #     text = t.format(
    #         prefix, name, primaries[0][0], primaries[0][1],
    #         " " * 8, primaries[1][0], primaries[1][1],
    #         " " * 8, primaries[2][0], primaries[2][1],
    #     )
    #     return text

    def make_primariey_info_text(self, prefix, name, primaries):
        t = "{} Gamut\n  name: {}\n  xy  : [{}, {}], [{}, {}], [{}, {}]"
        text = t.format(
            prefix, name, primaries[0][0], primaries[0][1],
            primaries[1][0], primaries[1][1],
            primaries[2][0], primaries[2][1],
        )
        return text

    def draw_information(self):
        """
        一度、sRGB相当で作り込んでから、Linearに変換する。
        """
        # 変数定義
        # --------------------------------
        fg_color = (font_level, font_level, font_level)
        bg_color = (0, 0, 0)
        font_size = self.int(self.get_img_width() * font_size_rate)
        text = self.make_info_text()

        # パラメータ計算
        # -------------------------------------------------
        width = self.diagram_width
        height = self.get_img_height() - self.diagram_height
        # テキストエリアの左上からのオフセット値
        h_offset = self.int(st_pos_offset * self.get_img_width())
        v_offset = h_offset
        # 背景画像に対するテキストエリアの開始座標
        st_pos = (self.get_img_width() - width, self.diagram_height)

        txt_img = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(txt_img)
        font = ImageFont.truetype(font_path, font_size)
        draw.text((h_offset, v_offset), text, font=font, fill=fg_color)
        text_img_sRGB = np.double(np.asarray(txt_img)) / 0xFF
        text_img_linear = tf.eotf(text_img_sRGB, tf.SRGB) * font_luminance

        tpg.merge(self.img, text_img_linear, st_pos)


if __name__ == '__main__':
    pass
