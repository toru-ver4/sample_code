# -*- coding: utf-8 -*-
"""
Compositeする
===================

"""

# import standard libraries
import os
import platform

# import third-party libraries
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import numpy as np

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
NOTO_SANS_CJKJP_MEDIUM\
    = "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Medium.otf"
NOTO_SANS_CJKJP_REGULAR\
    = "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf"
NOTO_SANS_CJKJP_BLACK\
    = "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Black.otf"
BIZUDGOTIC_BOLD\
    = "/usr/share/fonts/opentype/bizudp/BIZUDGothic-Bold.ttf"
BIZUDGOTIC_REGULAR\
    = "/usr/share/fonts/opentype/bizudp/BIZUDGothic-Regular.ttf"
BIZUD_P_GOTHIC_BOLD\
    = "/usr/share/fonts/opentype/bizudp/BIZUDPGothic-Bold.ttf"
BIZUD_P_GOTHIC_REGULAR\
    = "/usr/share/fonts/opentype/bizudp/BIZUDPGothic-Regular.ttf"

if platform.system() == "Windows":
    NOTO_SANS_CJKJP_MEDIUM\
        = "C:/Users/toruv/AppData/Local/Microsoft/Windows"\
        + "/Fonts/NotoSansJP-Medium.otf"


def get_text_width_height(
        text="0", font_size=10, font_path=NOTO_SANS_MONO_BOLD,
        stroke_width=0, stroke_fill=None):
    """
    Find out the text-width and text-height of the specific strings.

    Parameters
    ----------
    text : strings
        A text.
    font_size : int
        font size
    font_path : strings
        font path. ex. NOTO_SANS_MONO_REGULAR
    stroke_width : int
        stroke width
    stroke_fill : list or tuple(float)
        A color value like [1.0, 0.0, 0.0].

    Example
    -------
    >>> width, height = self.get_text_size(
    >>>     text="0120-777-777", font_size=10, font_path=NOTO_SANS_MONO_BOLD,
    >>>     stroke_width=0, stroke_fill=[0.1, 0.5, 1.0])

    """
    print(f"stroke_width={stroke_width}")
    dummy_img_size = 4095
    pos = (0, 0)
    dummy_img = np.zeros((dummy_img_size, dummy_img_size, 3))
    text_draw_ctrl = TextDrawControl(
        text=text,
        font_color=(0xFF, 0xFF, 0xFF),
        font_size=font_size,
        font_path=font_path,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill)
    text_draw_ctrl.draw(img=dummy_img, pos=pos)
    text_width, text_height = text_draw_ctrl.get_text_size()
    return text_width, text_height


class TextDrawControl():
    def __init__(
            self, text="hoge", font_color=(1.0, 1.0, 0.0),
            font_size=30, font_path=NOTO_SANS_MONO_BOLD,
            stroke_width=0, stroke_fill=None):
        """
        テキストをプロットするクラスのコンストラクタ

        Parameters
        ----------
        text : strings
            text for drawing.
        font_color : list or tuple(float)
            font color. it must be a linear space.
        font_size : int
            font size
        stroke_width : int
            outline width of the text.
        stroke_fill : list or tuple(float)
            A color value like [1.0, 0.0, 0.0].

        Returns
        -------
        array_like
            image data with line.

        Examples
        --------
        >>> dst_img = np.ones((540, 960, 3)) * np.array([0.3, 0.3, 0.1])
        >>> text_drawer = TextDrawer(
        >>>     dst_img, text="天上天下唯我独尊", pos=(200, 50),
        >>>     font_color=(0.5, 0.5, 0.5), font_size=30,
        >>>     bg_transfer_functions=tf.SRGB)
        >>> text_drawer.draw()
        >>> img = text_drawer.get_img()
        >>> cv2.imwrite("hoge.png", np.uint8(np.round(img[:, :, ::-1] * 0xFF)))
        """
        self.text = text
        self.font_size = font_size
        # self.font_color = tuple(
        #     np.uint8(np.round(np.append(np.array(font_color), 1.0) * 0xFF)))
        maximum_color_value = np.max(font_color)
        if maximum_color_value > 1.0:
            self.hdr_gain = maximum_color_value
        else:
            self.hdr_gain = 1.0
        self.font_color = np.append(np.array(font_color)/self.hdr_gain, 1.0)
        self.font_color = tuple(
            np.uint8(np.round(tf.oetf(self.font_color, tf.SRGB) * 0xFF)))
        self.bg_color = tuple(
            np.array([0x00, 0x00, 0x00, 0x00], dtype=np.uint8))
        self.font_path = font_path
        self.stroke_width = stroke_width
        if stroke_fill is not None:
            self.stroke_fill = tf.oetf(np.array(stroke_fill), tf.SRGB) * 0xFF
            self.stroke_fill = tuple(np.uint8(np.round(self.stroke_fill)))
        else:
            self.stroke_fill = None

        # self.debug_dump_parameters()

    def debug_dump_parameters(self):
        print(f"font_color={self.font_color}")
        print(f"stroke_fill={self.stroke_fill}")
        print(f"hdr_gain={self.hdr_gain}")

    def get_text_width_height(self):
        # dummy_img_size = 4095
        dummy_img_size = 4095*2
        dummy_img = np.zeros((dummy_img_size, dummy_img_size, 3))
        self.draw(img=dummy_img, pos=(0, 0))
        width, height = self.get_text_size()

        return width, height

    def draw(self, img, pos=(0, 0)):
        """
        drww text on the background image.

        Parameters
        ----------
        img : array_like(float, **linear**)
            background image data. it must be a linear space.
        pos : list or tuple(int)
            text strat position.
        """
        self.img = img
        self.pos = pos
        self.make_text_img_with_alpha()
        self.composite_text()

    def draw_with_dropped_dot(
            self, img, pos=(0, 0), dot_factor=0, offset=(0, 0)):
        """
        draw a dot-dropped text

        Parameters
        ----------
        dot_factor : int
            dot_factor = 0: no drop
            dot_factor = 1: 1x1 px drop
            dot_factor = 2: 2x2 px drop
            dot_factor = 3: 4x4 px drop
        offset : touple of int
            A offset dot-drop starts.
        """
        self.img = img
        self.pos = pos
        self.make_text_img_with_alpha()
        self.drop_dot(dot_factor, offset)
        self.composite_text()

    def drop_dot(self, dot_factor=0, offset=(0, 0)):
        """
        do the dot-dropping process.

        Parameters
        ----------
        dot_factor : int
            dot_factor = 0: no drop
            dot_factor = 1: 1x1 px drop
            dot_factor = 2: 2x2 px drop
            dot_factor = 3: 4x4 px drop
        offset : touple of int
            A offset dot-drop starts.
        """
        mod_val = 2 ** dot_factor
        div_val = mod_val // 2
        v_idx_list = np.arange(self.text_img.shape[0]) + offset[1]
        h_idx_list = np.arange(self.text_img.shape[1]) + offset[0]
        idx_even = (v_idx_list % mod_val // div_val == 0)[:, np.newaxis]\
            * (h_idx_list % mod_val // div_val == 0)[np.newaxis, :]
        idx_odd = (v_idx_list % mod_val // div_val == 1)[:, np.newaxis]\
            * (h_idx_list % mod_val // div_val == 1)[np.newaxis, :]
        idx = idx_even | idx_odd
        self.text_img[idx] = 0.0

    def split_rgb_alpha_from_rgba_img(self, img):
        self.rgb_img = img[:, :, :3]
        self.alpha_img = np.dstack((img[:, :, 3], img[:, :, 3], img[:, :, 3]))

    def make_text_img_with_alpha(self):
        """
        draw text and split rgb image and alpha image.
        """
        dummy_img = Image.new("RGBA", (1, 1), self.bg_color)
        dummy_draw = ImageDraw.Draw(dummy_img)
        font = ImageFont.truetype(self.font_path, self.font_size)
        text_size = dummy_draw.textsize(
            self.text, font, stroke_width=self.stroke_width)
        (_, _), (_, offset_y) = font.font.getsize(self.text)

        text_img = Image.new(
            "RGBA", (text_size[0], text_size[1]), self.bg_color)
        draw = ImageDraw.Draw(text_img)
        font = ImageFont.truetype(self.font_path, self.font_size)
        draw.text(
            (self.stroke_width, self.stroke_width),
            self.text, font=font, fill=self.font_color,
            stroke_width=self.stroke_width, stroke_fill=self.stroke_fill)
        text_img = np.asarray(text_img)[offset_y:text_size[1]] / 0xFF
        self.text_img_linear = tf.eotf(text_img, tf.SRGB)

    def composite_text(self):
        text_width = self.text_img_linear.shape[1]
        text_height = self.text_img_linear.shape[0]

        img_width = self.img.shape[1]
        img_height = self.img.shape[0]
        if self.pos[0]+text_width > img_width:
            msg = "\n===================================\n"
            msg += "font_control2.py: Text width is too big.\n"
            msg += "==================================="
            raise ValueError(msg)
        if self.pos[1]+text_height > img_height:
            msg = "\n===================================\n"
            msg += "font_control2.py: Text height is too big.\n"
            msg += "==================================="
            raise ValueError(msg)

        composite_area_img = self.img[self.pos[1]:self.pos[1]+text_height,
                                      self.pos[0]:self.pos[0]+text_width]
        bg_img_linear = composite_area_img
        text_img_linear = self.text_img_linear
        alpha = text_img_linear[:, :, 3:]
        a_idx = (alpha > 0)[..., 0]

        bg_img_linear[a_idx] = (1 - alpha[a_idx])\
            * bg_img_linear[a_idx]\
            + (text_img_linear[a_idx, :-1] * alpha[a_idx] * self.hdr_gain)

        self.img[self.pos[1]:self.pos[1]+text_height,
                 self.pos[0]:self.pos[0]+text_width] = bg_img_linear

    def get_text_size(self):
        """
        Returns
        -------
        width, height : int
            width and height
        """
        return self.text_img_linear.shape[1], self.text_img_linear.shape[0]


def simple_test_noraml_draw():
    from test_pattern_generator2 import img_wirte_float_as_16bit_int

    # example 1 SDR text on SDR background
    # prepare parameters
    fname = "/work/src/draw_test_01.png"
    width = 1280
    height = 720
    bg_color_sRGB = np.array([163, 199, 253]) / 255
    bg_color = tf.eotf(bg_color_sRGB, tf.SRGB)
    fg_color_sRGB = np.array([96, 192, 0]) / 255
    fg_color = tf.eotf(fg_color_sRGB, tf.SRGB)
    dst_img = np.ones((height, width, 3)) * bg_color

    # create instance
    text_draw_ctrl = TextDrawControl(
        text="天上天下唯我独尊", font_color=fg_color,
        font_size=140, font_path=NOTO_SANS_CJKJP_MEDIUM,
        stroke_width=0, stroke_fill=None)

    # calc position
    text_width, text_height = text_draw_ctrl.get_text_width_height()
    pos_h = (width // 2) - (text_width // 2)
    pos_v = (height // 2) - (text_height // 2)
    pos = (pos_h, pos_v)

    text_draw_ctrl.draw(img=dst_img, pos=pos)

    dst_img_gm24 = tf.oetf(np.clip(dst_img, 0.0, 1.0), tf.SRGB)
    img_wirte_float_as_16bit_int(fname, dst_img_gm24)

    # example 2 SDR text on SDR background
    # prepare parameters
    fname = "/work/src/draw_test_02.png"
    bg_color_sRGB = np.array([64, 64, 64]) / 255
    bg_color = tf.eotf(bg_color_sRGB, tf.SRGB)
    fg_color_sRGB = np.array([255, 255, 255]) / 255
    fg_color = tf.eotf(fg_color_sRGB, tf.SRGB)
    dst_img = np.ones((height, width, 3)) * bg_color

    # create instance
    text_draw_ctrl = TextDrawControl(
        text="黒背景に白色はどうだろう？", font_color=fg_color,
        font_size=70, font_path=NOTO_SANS_CJKJP_MEDIUM,
        stroke_width=0, stroke_fill=None)

    # calc position
    text_width, text_height = text_draw_ctrl.get_text_width_height()
    pos_h = (width // 2) - (text_width // 2)
    pos_v = (height // 2) - (text_height // 2)
    pos = (pos_h, pos_v)

    text_draw_ctrl.draw(img=dst_img, pos=pos)

    dst_img_gm24 = tf.oetf(np.clip(dst_img, 0.0, 1.0), tf.SRGB)
    img_wirte_float_as_16bit_int(fname, dst_img_gm24)

    # example 3 SDR text on SDR background
    # prepare parameters
    width = 1280
    height = 720
    fname = "/work/src/draw_test_03.png"
    bg_color_sRGB = np.array([255, 255, 255]) / 255
    bg_color = tf.eotf(bg_color_sRGB, tf.SRGB)
    fg_color_sRGB = np.array([0, 0, 0]) / 255
    fg_color = tf.eotf(fg_color_sRGB, tf.SRGB)
    dst_img = np.ones((height, width, 3)) * bg_color

    # create instance
    text_draw_ctrl = TextDrawControl(
        text="白背景に黒色でございます", font_color=fg_color,
        font_size=70, font_path=NOTO_SANS_CJKJP_MEDIUM,
        stroke_width=0, stroke_fill=None)

    # calc position
    text_width, text_height = text_draw_ctrl.get_text_width_height()
    pos_h = (width // 2) - (text_width // 2)
    pos_v = (height // 2) - (text_height // 2)
    pos = (pos_h, pos_v)

    text_draw_ctrl.draw(img=dst_img, pos=pos)

    dst_img_gm24 = tf.oetf(np.clip(dst_img, 0.0, 1.0), tf.SRGB)
    img_wirte_float_as_16bit_int(fname, dst_img_gm24)


    # example 4 SDR text on SDR background
    # prepare parameters
    fname = "/work/src/draw_test_04.png"
    bg_color_sRGB = np.array([64, 64, 64]) / 255
    bg_color = tf.eotf(bg_color_sRGB, tf.SRGB)
    fg_color_sRGB = np.array([224, 224, 224]) / 255
    fg_color = tf.eotf(fg_color_sRGB, tf.SRGB)
    edge_color_sRGB = np.array([224, 0, 0]) / 255
    edge_color = tf.eotf(edge_color_sRGB, tf.SRGB)
    dst_img = np.ones((height, width, 3)) * bg_color

    # create instance
    text_draw_ctrl = TextDrawControl(
        text="エッジを付けてみた！", font_color=fg_color,
        font_size=70, font_path=NOTO_SANS_CJKJP_MEDIUM,
        stroke_width=10, stroke_fill=edge_color)

    # calc position
    text_width, text_height = text_draw_ctrl.get_text_width_height()
    pos_h = (width // 2) - (text_width // 2)
    pos_v = (height // 2) - (text_height // 2)
    pos = (pos_h, pos_v)

    text_draw_ctrl.draw(img=dst_img, pos=pos)

    dst_img_gm24 = tf.oetf(np.clip(dst_img, 0.0, 1.0), tf.SRGB)
    img_wirte_float_as_16bit_int(fname, dst_img_gm24)


def simple_test_hdr_draw():
    from test_pattern_generator2 import img_wirte_float_as_16bit_int

    # example 1 SDR text on SDR background
    # prepare parameters
    fname = "/work/src/draw_hdr_01.png"
    width = 1920
    height = 1080
    bg_color_st2084 = np.array([520, 520, 520]) / 1023
    bg_color = tf.eotf_to_luminance(bg_color_st2084, tf.ST2084) / 100
    fg_color_st2084 = np.array([769, 769, 769]) / 1023
    fg_color_1000 = tf.eotf_to_luminance(fg_color_st2084, tf.ST2084) / 100
    fg_color_st2084 = np.array([848, 848, 848]) / 1023
    fg_color_4000 = tf.eotf_to_luminance(fg_color_st2084, tf.ST2084) / 100
    fg_color_st2084 = np.array([1023, 1023, 1023]) / 1023
    fg_color_10000 = tf.eotf_to_luminance(fg_color_st2084, tf.ST2084) / 100
    dst_img = np.ones((height, width, 3)) * bg_color

    # create instance
    text_draw_ctrl = TextDrawControl(
        text="1000 nit だよ！", font_color=fg_color_1000,
        font_size=140, font_path=NOTO_SANS_CJKJP_MEDIUM,
        stroke_width=0, stroke_fill=None)
    # calc position
    text_width, text_height = text_draw_ctrl.get_text_width_height()
    pos_h = (width // 2) - (text_width // 2)
    pos_v = text_height // 2
    pos = (pos_h, pos_v)
    # draw
    text_draw_ctrl.draw(img=dst_img, pos=pos)

    # create instance
    text_draw_ctrl = TextDrawControl(
        text="4000 nit だよ！", font_color=fg_color_4000,
        font_size=140, font_path=NOTO_SANS_CJKJP_MEDIUM,
        stroke_width=0, stroke_fill=None)
    # calc position
    text_width, text_height = text_draw_ctrl.get_text_width_height()
    pos_h = (width // 2) - (text_width // 2)
    pos_v = (height // 2) - (text_height // 2)
    pos = (pos_h, pos_v)
    # draw
    text_draw_ctrl.draw(img=dst_img, pos=pos)

    # create instance
    text_draw_ctrl = TextDrawControl(
        text="10000 nit だよ！", font_color=fg_color_10000,
        font_size=140, font_path=NOTO_SANS_CJKJP_MEDIUM,
        stroke_width=0, stroke_fill=None)
    # calc position
    text_width, text_height = text_draw_ctrl.get_text_width_height()
    pos_h = (width // 2) - (text_width // 2)
    pos_v = height - text_height - (text_height // 2)
    pos = (pos_h, pos_v)
    # draw
    text_draw_ctrl.draw(img=dst_img, pos=pos)

    dst_img_gm24 = tf.oetf_from_luminance(
        np.clip(dst_img*100, 0.0, 10000), tf.ST2084)
    img_wirte_float_as_16bit_int(fname, dst_img_gm24)

    # example 1 SDR text on SDR background
    # prepare parameters
    fname = "/work/src/draw_hdr_02.png"
    width = 1920
    height = 1080
    bg_color_st2084 = np.array([769, 769, 769]) / 1023
    bg_color = tf.eotf_to_luminance(bg_color_st2084, tf.ST2084) / 100
    fg_color_st2084 = np.array([520, 520, 520]) / 1023
    fg_color_100 = tf.eotf_to_luminance(fg_color_st2084, tf.ST2084) / 100
    fg_color_st2084 = np.array([0, 0, 0]) / 1023
    fg_color_0 = tf.eotf_to_luminance(fg_color_st2084, tf.ST2084) / 100
    dst_img = np.ones((height, width, 3)) * bg_color

    # create instance
    text_draw_ctrl = TextDrawControl(
        text="1000 nit 背景に 100 nit", font_color=fg_color_100,
        font_size=140, font_path=NOTO_SANS_CJKJP_MEDIUM,
        stroke_width=0, stroke_fill=None)
    # calc position
    text_width, text_height = text_draw_ctrl.get_text_width_height()
    pos_h = (width // 2) - (text_width // 2)
    pos_v = text_height
    pos = (pos_h, pos_v)
    # draw
    text_draw_ctrl.draw(img=dst_img, pos=pos)

    # create instance
    text_draw_ctrl = TextDrawControl(
        text="小さい文字だと背景が眩しすぎてエッジが消えません？",
        font_color=fg_color_0,
        font_size=20, font_path=NOTO_SANS_CJKJP_MEDIUM,
        stroke_width=0, stroke_fill=None)
    # calc position
    text_width, text_height = text_draw_ctrl.get_text_width_height()
    pos_h = (width // 2) - (text_width // 2)
    pos_v = (height // 2) - (text_height // 2)
    pos = (pos_h, pos_v)
    # draw
    text_draw_ctrl.draw(img=dst_img, pos=pos)

    # create instance
    text_draw_ctrl = TextDrawControl(
        text="1000 nit 背景に 0 nit", font_color=fg_color_0,
        font_size=140, font_path=NOTO_SANS_CJKJP_MEDIUM,
        stroke_width=0, stroke_fill=None)
    # calc position
    text_width, text_height = text_draw_ctrl.get_text_width_height()
    pos_h = (width // 2) - (text_width // 2)
    pos_v = height - text_height * 2
    pos = (pos_h, pos_v)
    # draw
    text_draw_ctrl.draw(img=dst_img, pos=pos)

    dst_img_hdr = tf.oetf_from_luminance(
        np.clip(dst_img*100, 0.0, 10000), tf.ST2084)
    img_wirte_float_as_16bit_int(fname, dst_img_hdr)


# def simple_test_dot_drop(dot_factor=1, offset=(0, 0)):
#     # example 1x1 drop
#     dst_img = np.ones((300, 300, 3)) * np.array([0.2, 0.2, 0.2])
#     text_drawer = TextDrawer(
#         dst_img, text="■", pos=(0, 0),
#         font_color=(1., 1., 1.), font_size=150,
#         bg_transfer_functions=tf.SRGB,
#         fg_transfer_functions=tf.SRGB)
#     text_drawer.draw_with_dropped_dot(dot_factor=dot_factor, offset=offset)
#     img = text_drawer.get_img()
#     fname = "/work/overuse/2020/005_make_countdown_movie/"\
#         + f"dot_drop_factor-{dot_factor}_offset-{offset[0]}-{offset[1]}"\
#         + ".png"
#     cv2.imwrite(
#         fname,
#         np.uint8(np.round(img[:, :, ::-1] * 0xFF)))


# def simple_test_with_stroke(font_path):
#     # example 1 SDR text on SDR background
#     dst_img = np.ones((540, 960, 3)) * np.array([0.3, 0.3, 0.1])
#     text_drawer = TextDrawer(
#         dst_img, text="0123456", pos=(200, 50),
#         font_color=(0.5, 0.5, 0.5), font_size=40,
#         bg_transfer_functions=tf.SRGB,
#         fg_transfer_functions=tf.SRGB,
#         stroke_width=4,
#         stroke_fill='black',
#         font_path=font_path)
#     text_drawer.draw()
#     img = text_drawer.get_img()
#     cv2.imwrite(
#         "sdr_text_on_sdr_image_with_stroke.png",
#         np.uint8(np.round(img[:, :, ::-1] * 0xFF)))


def draw_udp_gothic():
    from test_pattern_generator2 import img_wirte_float_as_16bit_int
    # prepare parameters
    width = 1280
    height = 720
    fname = "/work/src/bizudp_03.png"
    bg_color_sRGB = np.array([255, 255, 255]) / 255
    bg_color = tf.eotf(bg_color_sRGB, tf.SRGB)
    fg_color_sRGB = np.array([0, 0, 0]) / 255
    fg_color = tf.eotf(fg_color_sRGB, tf.SRGB)
    dst_img = np.ones((height, width, 3)) * bg_color

    # create instance
    text_draw_ctrl = TextDrawControl(
        text="BIZUD GOTHIC を描いてみた",
        font_color=fg_color,
        font_size=80, font_path=BIZUDGOTIC_BOLD,
        stroke_width=0, stroke_fill=None)

    # calc position
    text_width, text_height = text_draw_ctrl.get_text_width_height()
    pos_h = (width // 2) - (text_width // 2)
    pos_v = (height // 2) - (text_height // 2)
    pos = (pos_h, pos_v)

    text_draw_ctrl.draw(img=dst_img, pos=pos)

    dst_img_gm24 = tf.oetf(np.clip(dst_img, 0.0, 1.0), tf.SRGB)
    img_wirte_float_as_16bit_int(fname, dst_img_gm24)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # draw_udp_gothic()
    # pass
    simple_test_noraml_draw()
    simple_test_hdr_draw()

    # width = 1280
    # height = 720
    # fname = "/work/src/draw_test_03.png"
    # bg_color_sRGB = np.array([0, 0, 0]) / 255
    # bg_color = tf.eotf(bg_color_sRGB, tf.SRGB)
    # fg_color_sRGB = np.array([192, 192, 192]) / 255
    # fg_color = tf.eotf(fg_color_sRGB, tf.SRGB)
    # dst_img = np.ones((height, width, 3)) * bg_color

    # # create instance
    # text_draw_ctrl = TextDrawControl(
    #     text="白背景に黒色でございます", font_color=fg_color,
    #     font_size=70, font_path=NOTO_SANS_CJKJP_MEDIUM,
    #     stroke_width=0, stroke_fill=None)

    # # calc position
    # text_width, text_height = text_draw_ctrl.get_text_width_height()
    # pos_h = (width // 2) - (text_width // 2)
    # pos_v = (height // 2) - (text_height // 2)
    # pos = (pos_h, pos_v)

    # text_draw_ctrl.draw(img=dst_img, pos=pos)

    # dst_img_gm24 = tf.oetf(np.clip(dst_img, 0.0, 1.0), tf.SRGB)
    # # from test_pattern_generator2 import img_wirte_float_as_16bit_int
    # # img_wirte_float_as_16bit_int(fname, dst_img_gm24)
