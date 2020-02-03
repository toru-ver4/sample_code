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
import test_pattern_generator2 as tpg
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


# FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansMonoCJKjp-Regular.otf"
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansMonoCJKjp-Bold.otf"


def get_text_img_with_alpha(text="hoge", font_size=40):
    """
    アルファチャンネル付きで画像を作成
    """
    text_width = 1920
    text_height = 1080

    fg_color = (0x80, 0x80, 0x80, 0xFF)
    bg_color = (0x00, 0x00, 0x00, 0x00)

    txt_img = Image.new("RGBA", (text_width, text_height), bg_color)
    draw = ImageDraw.Draw(txt_img)
    font = ImageFont.truetype(FONT_PATH, font_size)
    text_size = draw.textsize(text, font)
    ascent, descent = font.getmetrics()
    (width, baseline), (offset_x, offset_y) = font.font.getsize(text)
    text_height = ascent - offset_y
    print(f"ascent: {ascent}, offset_y: {offset_y}")

    draw.text((0, 0), text, font=font, fill=fg_color)
    txt_img = np.asarray(txt_img) / 0xFF

    return txt_img[offset_y:text_size[1], :text_size[0]]


def split_rgb_alpha_from_rgba_img(img):
    rgb_img = img[:, :, :3]
    alpha_img = np.dstack((img[:, :, 3], img[:, :, 3], img[:, :, 3]))

    return rgb_img, alpha_img


def scaling_image(src_img, ratio=8):
    dst_img = cv2.resize(src_img, dsize=None, fx=ratio, fy=ratio,
                         interpolation=cv2.INTER_NEAREST)

    return dst_img


def save_image(img, name="hoge.png"):
    cv2.imwrite(name, np.uint8(np.round(img * 0xFF)))


def experimental_func(text_img_size=(960, 540), font_size=40):
    # テキストイメージ作成
    text_img = get_text_img_with_alpha(text="如月", font_size=40)
    rgb_img, alpha_img = split_rgb_alpha_from_rgba_img(text_img)
    rgb_img = scaling_image(rgb_img, ratio=8)
    alpha_img = scaling_image(alpha_img, ratio=8)

    save_image(rgb_img, "./blog_img/rgb.png")
    save_image(alpha_img, "./blog_img/alpha.png")


def composite_text(bg_img, text_img):
    bg_img_linear = tf.eotf(bg_img, tf.SRGB)
    text_img_linear = tf.eotf(text_img, tf.SRGB)
    print(np.max(bg_img_linear), np.max(text_img_linear[:, :, :-1]))

    bg_img_max_value = np.max(bg_img_linear)

    alpha = text_img[:, :, 3:]

    bg_img_linear = (1 - alpha) * bg_img_linear * bg_img_max_value\
        + text_img_linear[:, :, :-1]
    dst_bg_img = tf.oetf(bg_img_linear, tf.SRGB)
    tpg.preview_image(scaling_image(dst_bg_img, ratio=8))


def composite_text_test():
    text_img = get_text_img_with_alpha(text="abcdefghijklmn", font_size=40)
    dst_img = np.ones((text_img[:, :, :-1].shape)) * np.array([0.3, 0.3, 0.1])
    composite_text(dst_img, text_img)


class TextDrawer():
    def __init__(
            self, img, text="hoge", pos=(0, 0), font_color=(1.0, 1.0, 0.0),
            font_size=30, transfer_functions=tf.SRGB):
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
        >>> import transfer_functions as tf
        >>> img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        >>> text_drawer = TextDrawer(
                img/0xFF, text="huga", pos=(0, 0), font_color=(0.0, 0.5, 1.0),
                font_size=10, transfer_functions=tf.ST2084)
        >>> draw_text.draw()
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

    def draw(self):
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
        font = ImageFont.truetype(FONT_PATH, self.font_size)
        text_size = dummy_draw.textsize(self.text, font)
        (_, _), (_, offset_y) = font.font.getsize(self.text)

        text_img = Image.new(
            "RGBA", (text_size[0], text_size[1]), self.bg_color)
        draw = ImageDraw.Draw(text_img)
        font = ImageFont.truetype(FONT_PATH, self.font_size)
        draw.text((0, 0), self.text, font=font, fill=self.font_color)
        self.text_img = np.asarray(text_img)[offset_y:text_size[1]] / 0xFF

    def composite_text(self):
        text_width = self.text_img.shape[1]
        text_height = self.text_img.shape[0]
        composite_area_img = self.img[self.pos[1]:self.pos[1]+text_height,
                                      self.pos[0]:self.pos[0]+text_width]
        bg_img_linear = tf.eotf(composite_area_img, self.tf)
        text_img_linear = tf.eotf(self.text_img, tf.SRGB)
        print(np.max(bg_img_linear), np.max(text_img_linear[:, :, :-1]))

        alpha = self.text_img[:, :, 3:]

        bg_img_linear = (1 - alpha) * bg_img_linear\
            + text_img_linear[:, :, :-1]
        bg_img_linear = tf.oetf(bg_img_linear, tf.SRGB)
        self.img[self.pos[1]:self.pos[1]+text_height,
                 self.pos[0]:self.pos[0]+text_width] = bg_img_linear
        tpg.preview_image(scaling_image(self.img, ratio=2))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # experimental_func()
    # composite_text_test()

    dst_img = np.ones((540, 960, 3)) * np.array([0.3, 0.3, 0.1])
    text_drawer = TextDrawer(
        dst_img, text="天上天下唯我独尊", pos=(0, 0),
        font_color=(0.5, 0.5, 0.5, 1.0), font_size=20,
        transfer_functions=tf.SRGB)
    text_drawer.draw()
