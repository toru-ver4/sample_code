# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
from multiprocessing.sharedctypes import Value
import os

# import third-party libraries
import numpy as np
from colour.io import write_image

# import my libraries
import font_control2 as fc2
import test_pattern_generator2 as tpg
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


FFMPEG_NORMALIZE_COEF = 65340
FFMPEG_NORMALIZE_COEF_INV = 65535/65340


def conv_Nbit_to_linear(x, bit_depth=8, tf_str=tf.GAMMA24):
    """
    Examples
    --------
    >>> x = [255, 128, 64]
    >>> bit_depth = 8
    >>> tf_str = tf.ST2084
    >>> y = conv_8bit_to_linear(x, bit_depth, tf_str)
    >>> x2 = conv_linear_to_8bit(y, bit_depth, tf_str)
    >>> print(y)
    [  1.00000000e+02   9.40745992e-01   5.22570128e-02]
    >>> print(x2)
    [255 128  64]
    """
    max_value = (2 ** bit_depth) - 1
    y = tf.eotf_to_luminance(np.array(x)/max_value, tf_str)\
        / tf.REF_WHITE_LUMINANCE
    return y


def conv_linar_to_Nbit(x, bit_depth=8, tf_str=tf.GAMMA24):
    """
    Examples
    --------
    >>> x = [255, 128, 64]
    >>> bit_depth = 8
    >>> tf_str = tf.ST2084
    >>> y = conv_8bit_to_linear(x, bit_depth, tf_str)
    >>> x2 = conv_linear_to_8bit(y, bit_depth, tf_str)
    >>> print(y)
    [  1.00000000e+02   9.40745992e-01   5.22570128e-02]
    >>> print(x2)
    [255 128  64]
    """
    max_value = (2 ** bit_depth) - 1
    y = tf.oetf_from_luminance(x * tf.REF_WHITE_LUMINANCE, tf_str)
    y = np.round(y * max_value)

    if bit_depth <= 8:
        y_Nbit = np.uint8(y)
    elif bit_depth <= 16:
        y_Nbit = np.uint16(y)
    elif bit_depth <= 32:
        y_Nbit = np.uint32(y)
    else:
        y_Nbit = np.uint64(y)

    return y_Nbit


def conv_8bit_to_linear(x, tf_str=tf.GAMMA24):
    """
    Examples
    --------
    >>> x = [255, 128, 64]
    >>> tf_str = tf.ST2084
    >>> y = conv_8bit_to_linear(x, tf_str)
    >>> x2 = conv_linear_to_8bit(y, tf_str)
    >>> print(y)
    [  1.00000000e+02   9.40745992e-01   5.22570128e-02]
    >>> print(x2)
    [255 128  64]
    """
    y = conv_Nbit_to_linear(x, bit_depth=8, tf_str=tf_str)
    return y


def conv_linear_to_8bit(x, tf_str=tf.GAMMA24):
    """
    Examples
    --------
    >>> x = [255, 128, 64]
    >>> tf_str = tf.ST2084
    >>> y = conv_8bit_to_linear(x, tf_str)
    >>> x2 = conv_linear_to_8bit(y, tf_str)
    >>> print(y)
    [  1.00000000e+02   9.40745992e-01   5.22570128e-02]
    >>> print(x2)
    [255 128  64]
    """
    y = conv_linar_to_Nbit(x, bit_depth=8, tf_str=tf_str)

    return y


class LineUnit():
    """
    Methods
    -------
    * 
    """
    def __init__(
            self, width, height, line_width, num_of_lines,
            line_color1, line_color2, bg_color):
        """
        Parameters
        ----------
        width : int
            width
        height : int
            height
        line_width : int
            width of the line.
        num_of_lines : int
            number of the lines.
        line_color1 : ndarray
            line color value. It must be linear.
            (ex. np.array([0.5, 1.0, 0.9]))
        line_color2 : ndarray
            line color value. It must be linear.
            (ex. np.array([0.5, 1.0, 0.9]))
        bg_color : ndarray
            bg color value. It must be linear.
            (ex. np.array([0.5, 1.0, 0.9]))
        """
        self.width = width
        self.height = height
        self.line_width = line_width
        self.line_color1 = line_color1
        self.line_color2 = line_color2
        self.line_color_num = 2  # color1 and color2
        self.bg_color = bg_color
        self.init_parameter_check()

        self.create_bg_image()
        self.num_of_lines = num_of_lines
        self.calc_line_st_pos()
        self.draw()

    def calc_line_st_pos(self):
        line_area_width\
            = self.line_width * self.num_of_lines * self.line_color_num
        line_st_pos_h = self.width//2 - line_area_width//2
        line_st_pos_v = 0
        self.line_st_pos = [line_st_pos_h, line_st_pos_v]

    def init_parameter_check(self):
        if len(self.line_color1) != len(self.line_color2):
            msg = "line_color1 and line_color2 is not same lnegth."
            raise ValueError(msg)
        if len(self.line_color1) != len(self.bg_color):
            msg = "line_color1 and bg_color is not same lnegth."
            raise ValueError(msg)

    def create_bg_image(self):
        self.img = np.ones((self.height, self.width, 3)) * self.bg_color

    def draw(self):
        st_pos_h_base = self.line_st_pos[0]
        st_pos_v_base = self.line_st_pos[1]
        for idx in range(self.num_of_lines):
            st_pos_h\
                = st_pos_h_base + idx * self.line_width * self.line_color_num
            ed_pos_h = st_pos_h + self.line_width
            st_pos_v = st_pos_v_base
            ed_pos_v = st_pos_v_base + self.height
            self.img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h] = self.line_color1

            st_pos_h += self.line_width
            ed_pos_h += self.line_width
            self.img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h] = self.line_color2


def debug_line_unit():
    width = 640
    height = 360
    line_width = 4
    num_of_lines = 5
    line_color1 = conv_8bit_to_linear([255, 128, 64])
    line_color2 = conv_8bit_to_linear([0, 0, 0])
    bg_color = conv_8bit_to_linear([64, 64, 64])
    lu = LineUnit(
        width=width, height=height, line_width=line_width,
        num_of_lines=num_of_lines,
        line_color1=line_color1, line_color2=line_color2, bg_color=bg_color)
    img_linear = lu.img

    out_img = conv_linear_to_8bit(img_linear)
    write_image(out_img/0xFF, "./hoge.png", 'uint8')


def debug_func():
    debug_line_unit()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_func()
    # x = [255, 128, 64]
    # tf_str = tf.ST2084
    # y = conv_8bit_to_linear(x, tf_str)
    # print(y)
    # x2 = conv_linear_to_8bit(y, tf_str)
    # print(x2)

    # x = [1023, 512, 128]
    # tf_str = tf.ST2084
    # y = conv_Nbit_to_linear(x, bit_depth=10, tf_str=tf_str)
    # print(y)
    # x2 = conv_linar_to_Nbit(y, bit_depth=10, tf_str=tf_str)
    # print(x2)
