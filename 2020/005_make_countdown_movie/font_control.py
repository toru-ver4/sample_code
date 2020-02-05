# -*- coding: utf-8 -*-
"""
Compositeする
===================

"""

# import standard libraries
import os

# import third-party libraries
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import numpy as np
import cv2

# import my libraries
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


NOTO_SANS_MONO_REGULAR\
    = "/usr/share/fonts/opentype/noto/NotoSansMonoCJKjp-Regular.otf"
NOTO_SANS_MONO_BOLD\
    = "/usr/share/fonts/opentype/noto/NotoSansMonoCJKjp-Bold.otf"


class TextDrawer():
    def __init__(
            self, img, text="hoge", pos=(0, 0), font_color=(1.0, 1.0, 0.0),
            font_size=30, transfer_functions=tf.SRGB,
            font_path=NOTO_SANS_MONO_BOLD):
        """
        テキストをプロットするクラスのコンストラクタ

        Parameters
        ----------
        img : array_like
            image data
        text : strings
            text.
        pos : list or tuple(int)
            text position.
        font_color : list or tuple(float)
            font color.
        font_size : int
            font size
        transfer_function : strings
            transfer function

        Returns
        -------
        array_like
            image data with line.

        Examples
        --------
        >>> dst_img = np.ones((540, 960, 3)) * np.array([0.3, 0.3, 0.1])
        >>> text_drawer = TextDrawer(
        >>>     dst_img, text="天上天下唯我独尊", pos=(200, 50),
        >>>     font_color=(0.5, 0.5, 0.5, 1.0), font_size=30,
        >>>     transfer_functions=tf.SRGB)
        >>> text_drawer.draw()
        >>> img = text_drawer.get_img()
        >>> cv2.imwrite("hoge.png", np.uint8(np.round(img[:, :, ::-1] * 0xFF)))
        """
        self.img = img
        self.text = text
        self.pos = pos
        self.font_size = font_size
        self.font_color = tuple(
            np.uint8(np.round(np.array(font_color) * 0xFF)))
        self.bg_color = tuple(
            np.array([0x00, 0x00, 0x00, 0x00], dtype=np.uint8))
        self.tf = transfer_functions
        self.NOTO_SANS_MONO_BOLD = NOTO_SANS_MONO_BOLD

    def draw(self):
        """
        テキストを描画する。
        """
        self.make_text_img_with_alpha()
        self.composite_text()

    def split_rgb_alpha_from_rgba_img(self, img):
        self.rgb_img = img[:, :, :3]
        self.alpha_img = np.dstack((img[:, :, 3], img[:, :, 3], img[:, :, 3]))

    def make_text_img_with_alpha(self):
        """
        アルファチャンネル付きで画像を作成
        """
        dummy_img = Image.new("RGBA", (1, 1), self.bg_color)
        dummy_draw = ImageDraw.Draw(dummy_img)
        font = ImageFont.truetype(self.NOTO_SANS_MONO_BOLD, self.font_size)
        text_size = dummy_draw.textsize(self.text, font)
        (_, _), (_, offset_y) = font.font.getsize(self.text)

        text_img = Image.new(
            "RGBA", (text_size[0], text_size[1]), self.bg_color)
        draw = ImageDraw.Draw(text_img)
        font = ImageFont.truetype(self.NOTO_SANS_MONO_BOLD, self.font_size)
        draw.text((0, 0), self.text, font=font, fill=self.font_color)
        self.text_img = np.asarray(text_img)[offset_y:text_size[1]] / 0xFF

    def composite_text(self):
        text_width = self.text_img.shape[1]
        text_height = self.text_img.shape[0]
        composite_area_img = self.img[self.pos[1]:self.pos[1]+text_height,
                                      self.pos[0]:self.pos[0]+text_width]
        bg_img_linear = tf.eotf_to_luminance(composite_area_img, self.tf)
        text_img_linear = tf.eotf_to_luminance(self.text_img, tf.SRGB)

        alpha = self.text_img[:, :, 3:]

        bg_img_linear = (1 - alpha) * bg_img_linear\
            + text_img_linear[:, :, :-1]
        bg_img_linear = tf.oetf_from_luminance(bg_img_linear, self.tf)
        self.img[self.pos[1]:self.pos[1]+text_height,
                 self.pos[0]:self.pos[0]+text_width] = bg_img_linear

    def get_img(self):
        return self.img


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # example
    dst_img = np.ones((540, 960, 3)) * np.array([0.3, 0.3, 0.1])
    text_drawer = TextDrawer(
        dst_img, text="天上天下唯我独尊", pos=(200, 50),
        font_color=(0.5, 0.5, 0.5, 1.0), font_size=40,
        transfer_functions=tf.SRGB)
    text_drawer.draw()
    img = text_drawer.get_img()
    cv2.imwrite("hoge.png", np.uint8(np.round(img[:, :, ::-1] * 0xFF)))
